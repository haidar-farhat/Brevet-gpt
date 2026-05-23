import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { BrevetEvent, BrevetService } from './brevet.service';

interface LogLine {
  stage: string;
  level: string;
  text: string;
}

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  // Inputs (signals so the zoneless app reacts to changes).
  readonly question = signal('');
  readonly language = signal<'auto' | 'en' | 'fr'>('auto');
  readonly subject = signal('');

  // Streaming state.
  readonly busy = signal(false);
  readonly logs = signal<LogLine[]>([]);
  readonly answer = signal('');
  readonly citations = signal<any[]>([]);
  readonly metrics = signal<any | null>(null);
  readonly routed = signal<{ language?: string; subject?: string | null }>({});

  constructor(private readonly api: BrevetService) {}

  onEnter(event: Event): void {
    const ke = event as KeyboardEvent;
    if (!ke.shiftKey) {
      event.preventDefault();
      this.ask();
    }
  }

  async ask(): Promise<void> {
    const q = this.question().trim();
    if (!q || this.busy()) return;

    this.busy.set(true);
    this.logs.set([]);
    this.answer.set('');
    this.citations.set([]);
    this.metrics.set(null);
    this.routed.set({});
    this.pushLog('you', 'you', q);

    const lang = this.language() === 'auto' ? null : this.language();
    const subject = this.subject().trim() || null;

    await this.api.ask(q, lang, subject, {
      onEvent: (ev) => this.handleEvent(ev),
      onError: (m) => this.pushLog('error', 'error', m),
      onDone: () => this.busy.set(false),
    });
  }

  private handleEvent(ev: BrevetEvent): void {
    switch (ev.type) {
      case 'log':
        this.handleLog(ev);
        break;
      case 'token':
        this.answer.update((a) => a + (ev['text'] ?? ''));
        break;
      case 'result':
        this.citations.set(ev['citations'] ?? []);
        this.metrics.set(ev['metrics'] ?? null);
        if (ev['refused'] || !this.answer()) this.answer.set(ev['answer'] ?? this.answer());
        this.pushLog('done', 'ok', this.metricsLine(ev['metrics']));
        break;
      case 'error':
        this.pushLog('error', 'error', ev['error'] ?? 'unknown error');
        break;
    }
  }

  private handleLog(ev: BrevetEvent): void {
    if (ev['stage'] === 'route') {
      this.routed.set({ language: ev['language'], subject: ev['subject'] });
      const queries = (ev['queries'] ?? []).join('  |  ');
      this.pushLog('route', 'info',
        `lang=${ev['language']}  subject=${ev['subject'] ?? '—'}  (${ev['latency_s']}s)\n    queries: ${queries}`);
    } else if (ev['stage'] === 'retrieve' && ev['sources']) {
      this.pushLog('retrieve', 'info',
        `${ev['chunks']} chunks · best sim ${ev['best_sim']} · ${ev['latency_s']}s · reforms ${ev['reformulations']}`);
      for (const s of ev['sources']) {
        this.pushLog('retrieve', 'dim', `    [${s.n}] ${s.book} p.${s.page}  (sim ${s.sim})`);
      }
    } else {
      this.pushLog(ev['stage'] ?? 'log', ev['level'] ?? 'info', ev['message'] ?? '');
    }
  }

  private metricsLine(m: any): string {
    if (!m) return 'done';
    const l = m.latency ?? {};
    return `model ${m.model} · ${m.total_tokens} tok · ${m.generation_tokens_per_sec} tok/s · ` +
      `retrieve ${l.retrieve_s}s · gen ${l.generate_s}s · total ${l.total_s}s · ` +
      `ctx ${m.context_chunks} · sim ${m.best_similarity}`;
  }

  private pushLog(stage: string, level: string, text: string): void {
    this.logs.update((ls) => [...ls, { stage, level, text }]);
  }
}
