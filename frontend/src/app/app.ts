import { Component, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

import { BrevetEvent, BrevetService } from './brevet.service';
import { renderRich } from './format';
import { ManageComponent } from './manage.component';
import { MaterialsService } from './materials.service';

interface LogLine {
  stage: string;
  level: string;
  text: string;
}

const STAGE_LABEL: Record<string, string> = {
  guard: 'Checking question',
  route: 'Routing',
  retrieve: 'Searching textbooks',
  rerank: 'Re-ranking results',
  grade: 'Analysing context',
  refine: 'Refining the search',
  reason: 'Reasoning',
  solve: 'Solving step by step',
  generate: 'Writing the answer',
  verify: 'Verifying answer',
  answer: 'Finishing',
  done: 'Done',
};

const EXAMPLES = [
  { q: 'What is a mole in chemistry?', language: 'en', subject: 'chemistry' },
  { q: 'How does the human eye form an image?', language: 'en', subject: 'physics' },
  { q: 'Énonce le théorème de Pythagore.', language: 'fr', subject: 'math' },
  { q: "Qu'est-ce qu'un neurone ?", language: 'fr', subject: 'biology' },
];

@Component({
  selector: 'app-root',
  imports: [FormsModule, ManageComponent],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly subjects = signal<{ code: string; name_en: string; name_fr: string }[]>([]);
  readonly examples = EXAMPLES;

  // Which screen is showing: the study assistant or the materials manager.
  readonly view = signal<'study' | 'manage'>('study');
  private abort: AbortController | null = null;

  // Inputs
  readonly question = signal('');
  readonly language = signal<'auto' | 'en' | 'fr' | 'ar'>('auto');
  readonly subject = signal('');
  readonly grade = signal('');                                   // '' = all grades
  readonly grades = signal<{ code: string; name: string }[]>([]);

  // Run state
  readonly busy = signal(false);
  readonly logs = signal<LogLine[]>([]);
  readonly thinking = signal('');          // streamed reasoning text
  readonly thinkingOpen = signal(true);
  readonly answer = signal('');
  readonly refused = signal(false);
  readonly status = signal('answer');
  readonly citations = signal<any[]>([]);
  readonly contexts = signal<any[]>([]);
  readonly expanded = signal<Set<number>>(new Set());
  readonly metrics = signal<any | null>(null);
  readonly routed = signal<{ language?: string; subject?: string | null }>({});
  readonly stage = signal('');
  readonly copied = signal(false);

  readonly cached = computed(() => !!this.metrics()?.['cached']);

  readonly stageLabel = computed(() => STAGE_LABEL[this.stage()] ?? this.stage());
  readonly hasRun = computed(() => this.busy() || !!this.answer() || !!this.metrics());

  /** Headline performance figures. */
  readonly perf = computed(() => {
    const m = this.metrics();
    if (!m) return [];
    const l = m.latency ?? {};
    return [
      { label: 'Total time', value: `${l.total_s ?? '—'}s` },
      { label: 'Speed', value: `${m.generation_tokens_per_sec ?? 0} tok/s` },
      { label: 'Tokens', value: `${m.total_tokens ?? 0}`, hint: `${m.prompt_tokens ?? 0} in · ${m.completion_tokens ?? 0} out` },
      { label: 'LLM calls', value: `${m.llm_calls ?? 0}` },
      { label: 'Sources used', value: `${m.context_chunks ?? 0}` },
      { label: 'Top match', value: m.best_similarity != null ? Number(m.best_similarity).toFixed(2) : '—' },
    ];
  });

  /** Latency breakdown segments for the timing bar. */
  readonly latencyBars = computed(() => {
    const l = this.metrics()?.latency;
    const total = l?.total_s || 0;
    if (!total) return [];
    const route = l.reformulate_s || 0;
    const retrieve = l.retrieve_s || 0;
    const gen = l.generate_s || 0;
    const other = Math.max(0, total - route - retrieve - gen);
    const seg = (label: string, s: number, cls: string) =>
      ({ label, s: +s.toFixed(2), pct: Math.max(0, Math.round((s / total) * 100)), cls });
    return [seg('route', route, 's1'), seg('retrieve', retrieve, 's2'),
            seg('reason/verify', other, 's3'), seg('generate', gen, 's4')].filter((x) => x.s > 0);
  });

  /** Agentic-only quality badges (present when the agentic pipeline ran). */
  readonly quality = computed(() => {
    const m = this.metrics();
    if (!m || !m.agentic) return null;
    return {
      faithfulness: m.faithfulness,
      sufficient: m.context_sufficient,
      relevant: m.relevant_fraction,
      loops: m.loops,
      reranker: m.rerank_backend,
      revised: m.revised,
    };
  });

  constructor(
    private readonly api: BrevetService,
    private readonly sanitizer: DomSanitizer,
    private readonly materials: MaterialsService,
  ) {
    // Populate the subject + grade selectors from the catalog taxonomy (best-effort).
    this.materials.taxonomy()
      .then((t) => { this.subjects.set(t.subjects); this.grades.set(t.grades); })
      .catch(() => {});
  }

  /** Final answer rendered as Markdown + LaTeX (used once streaming finishes). */
  renderedAnswer(): SafeHtml {
    return this.sanitizer.bypassSecurityTrustHtml(renderRich(this.answer()));
  }

  tone(value: number | null | undefined): string {
    if (value == null) return 'muted';
    return value >= 0.7 ? 'good' : value >= 0.5 ? 'ok' : 'bad';
  }

  /** Localized display name for a subject (falls back to its code). */
  subjectLabel(s: { code: string; name_en: string; name_fr: string }): string {
    return (this.language() === 'fr' ? s.name_fr : s.name_en) || s.code;
  }

  /** Localized display name for a subject by code (citations / routed pill). */
  subjectName(code: string | null | undefined): string {
    if (!code) return '';
    const s = this.subjects().find((x) => x.code === code);
    return s ? this.subjectLabel(s) : code;
  }

  answerTitle(): string {
    switch (this.status()) {
      case 'clarify': return 'I need a bit more info';
      case 'out_of_scope': return 'Off-topic';
      case 'refused':
      case 'blocked': return 'Not in the materials';
      default: return 'Answer';
    }
  }

  answerClass(): string {
    const s = this.status();
    if (s === 'clarify') return 'clarify';
    if (s === 'out_of_scope' || s === 'refused' || s === 'blocked') return 'refused';
    return '';
  }

  isExpanded(n: number): boolean {
    return this.expanded().has(n);
  }

  toggleSource(n: number): void {
    const s = new Set(this.expanded());
    s.has(n) ? s.delete(n) : s.add(n);
    this.expanded.set(s);
  }

  toggleAllSources(): void {
    const all = this.contexts().map((c) => c.n);
    this.expanded.set(this.expanded().size >= all.length ? new Set() : new Set(all));
  }

  pct(value: number | null | undefined): string {
    return value == null ? '—' : `${Math.round(value * 100)}%`;
  }

  setExample(ex: { q: string; language: string; subject: string }): void {
    if (this.busy()) return;
    this.question.set(ex.q);
    this.language.set(ex.language as 'auto' | 'en' | 'fr' | 'ar');
    this.subject.set(ex.subject);
  }

  onEnter(event: Event): void {
    const ke = event as KeyboardEvent;
    if (!ke.shiftKey) {
      event.preventDefault();
      this.ask();
    }
  }

  async copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(this.answer());
      this.copied.set(true);
      setTimeout(() => this.copied.set(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  }

  async ask(): Promise<void> {
    const q = this.question().trim();
    if (!q || this.busy()) return;

    this.busy.set(true);
    this.logs.set([]);
    this.thinking.set('');
    this.thinkingOpen.set(true);
    this.answer.set('');
    this.refused.set(false);
    this.status.set('answer');
    this.citations.set([]);
    this.contexts.set([]);
    this.expanded.set(new Set());
    this.metrics.set(null);
    this.stage.set('route');

    const lang = this.language() === 'auto' ? null : this.language();
    const subject = this.subject().trim() || null;
    const grade = this.grade() || null;
    this.abort = new AbortController();

    await this.api.ask(q, lang, subject, {
      onEvent: (ev) => this.handleEvent(ev),
      onError: (m) => this.pushLog('error', 'error', m),
      onDone: () => {
        this.busy.set(false);
        this.stage.set('done');
        this.abort = null;
      },
    }, this.abort.signal, grade);
  }

  /** Abort an in-flight query (the SSE fetch is cancelled via AbortSignal). */
  stop(): void {
    if (!this.busy()) return;
    this.abort?.abort();
    this.abort = null;
    this.busy.set(false);
    this.stage.set('done');
    this.pushLog('answer', 'warn', 'stopped by user');
  }

  private handleEvent(ev: BrevetEvent): void {
    switch (ev.type) {
      case 'log':
        this.handleLog(ev);
        break;
      case 'reason_token':
        this.thinking.update((t) => t + (ev['text'] ?? ''));
        break;
      case 'token':
        this.answer.update((a) => a + (ev['text'] ?? ''));
        break;
      case 'result':
        this.refused.set(!!ev['refused']);
        this.status.set(ev['status'] ?? (ev['refused'] ? 'refused' : 'answer'));
        this.citations.set(ev['citations'] ?? []);
        this.contexts.set(ev['contexts'] ?? []);
        this.metrics.set(ev['metrics'] ?? null);
        this.routed.set({ language: ev['language'], subject: ev['subject'] });
        if (ev['refused'] || ev['status'] !== 'answer' || !this.answer()) {
          this.answer.set(ev['answer'] ?? this.answer());
        }
        break;
      case 'error':
        this.pushLog('error', 'error', ev['error'] ?? 'unknown error');
        break;
    }
  }

  private handleLog(ev: BrevetEvent): void {
    if (ev['stage']) this.stage.set(ev['stage']);
    if (ev['stage'] === 'route') {
      this.routed.set({ language: ev['language'], subject: ev['subject'] });
      const queries = (ev['queries'] ?? []).join('  •  ');
      this.pushLog('route', 'info',
        `lang=${ev['language']}  subject=${ev['subject'] ?? '—'}  (${ev['latency_s']}s)\n    queries: ${queries}`);
    } else if (ev['stage'] === 'retrieve' && ev['sources']) {
      this.pushLog('retrieve', 'info',
        `${ev['chunks']} chunks · best sim ${ev['best_sim']} · reforms ${ev['reformulations']}`);
      for (const s of ev['sources']) {
        this.pushLog('retrieve', 'dim', `    [${s.n}] ${s.book} p.${s.page}  (sim ${s.sim})`);
      }
    } else {
      this.pushLog(ev['stage'] ?? 'log', ev['level'] ?? 'info', ev['message'] ?? '');
    }
  }

  private pushLog(stage: string, level: string, text: string): void {
    this.logs.update((ls) => [...ls, { stage, level, text }]);
  }
}
