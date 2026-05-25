import { Component, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { DupMatch, MaterialBook, MaterialsService, UploadEvent } from './materials.service';

interface UpLog {
  stage: string;
  level: string;
  text: string;
}

const SUBJECTS = ['math', 'physics', 'chemistry', 'biology', 'informatics', 'grammar', 'reading', 'french', 'english'];

const UP_STAGE: Record<string, string> = {
  validate: 'Validating',
  parse: 'Extracting text',
  ocr: 'Running OCR',
  chunk: 'Chunking',
  embed: 'Embedding',
  store: 'Saving',
  done: 'Done',
};

@Component({
  selector: 'app-manage',
  imports: [FormsModule],
  templateUrl: './manage.component.html',
  styleUrl: './manage.component.css',
})
export class ManageComponent {
  readonly subjects = SUBJECTS;

  // Library list + filters
  readonly books = signal<MaterialBook[]>([]);
  readonly loading = signal(false);
  readonly ingestBusy = signal(false);
  readonly fStatus = signal('');
  readonly fSubject = signal('');
  readonly fLang = signal('');
  readonly fQuery = signal('');

  // Upload form
  readonly file = signal<File | null>(null);
  readonly fileName = signal('');
  readonly upTitle = signal('');
  readonly upLang = signal<'en' | 'fr'>('en');
  readonly upSubject = signal('');
  readonly upLevel = signal('brevet');

  // Upload run state
  readonly uploading = signal(false);
  readonly upLogs = signal<UpLog[]>([]);
  readonly upStage = signal('');
  readonly upDone = signal<any | null>(null);
  readonly pending = signal<DupMatch | null>(null);

  // Browse + corpus search
  readonly openBook = signal<any | null>(null);
  readonly searchQ = signal('');
  readonly searchHits = signal<any[] | null>(null);
  readonly searching = signal(false);

  readonly upStageLabel = computed(() => UP_STAGE[this.upStage()] ?? this.upStage());
  readonly canUpload = computed(
    () => !!this.file() && !!this.upTitle().trim() && !!this.upSubject() && !this.uploading(),
  );

  constructor(private readonly api: MaterialsService) {
    this.refresh();
  }

  // ---------------- library ----------------
  async refresh(): Promise<void> {
    this.loading.set(true);
    try {
      const res = await this.api.list({
        status: this.fStatus() || null,
        subject: this.fSubject() || null,
        language: this.fLang() || null,
        q: this.fQuery().trim() || null,
      });
      this.books.set(res.books);
      this.ingestBusy.set(res.ingest_in_progress);
    } catch {
      this.books.set([]);
    } finally {
      this.loading.set(false);
    }
  }

  async freeze(b: MaterialBook): Promise<void> {
    await this.api.setStatus(b.id, b.status === 'frozen' ? 'unfreeze' : 'freeze');
    this.refresh();
  }

  async del(b: MaterialBook): Promise<void> {
    if (!confirm(`Delete "${b.title}" and its ${b.chunks} chunks permanently?`)) return;
    await this.api.remove(b.id);
    if (this.openBook()?.id === b.id) this.openBook.set(null);
    this.refresh();
  }

  // ---------------- upload ----------------
  onFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const f = input.files?.[0] ?? null;
    this.file.set(f);
    this.fileName.set(f?.name ?? '');
    if (f && !this.upTitle().trim()) {
      this.upTitle.set(f.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim());
    }
  }

  async startUpload(): Promise<void> {
    if (!this.canUpload()) return;
    // Cheap metadata dedup check first (no file sent) so we can offer a choice.
    const match = await this.api.check({
      title: this.upTitle().trim(),
      language: this.upLang(),
      subject: this.upSubject(),
    });
    if (match) {
      this.pending.set(match);
      return;
    }
    this.doUpload(null, null);
  }

  resolve(resolution: 'update' | 'freeze_replace'): void {
    const m = this.pending();
    this.pending.set(null);
    this.doUpload(resolution, m?.book_id ?? null);
  }

  cancelResolve(): void {
    this.pending.set(null);
  }

  private doUpload(resolution: string | null, targetId: number | null): void {
    const f = this.file();
    if (!f) return;
    this.uploading.set(true);
    this.upLogs.set([]);
    this.upDone.set(null);
    this.upStage.set('validate');

    this.api.upload(
      f,
      {
        title: this.upTitle().trim(),
        language: this.upLang(),
        subject: this.upSubject(),
        level: this.upLevel().trim() || 'brevet',
        resolution,
        target_id: targetId,
      },
      {
        onEvent: (ev) => this.handleUpload(ev),
        onError: (msg) => this.pushUp('error', 'error', msg),
        onDone: () => {
          this.uploading.set(false);
          this.upStage.set('done');
          this.refresh();
        },
      },
    );
  }

  private handleUpload(ev: UploadEvent): void {
    switch (ev.type) {
      case 'stage':
        this.upStage.set(ev['stage']);
        this.pushUp(ev['stage'], 'info', ev['message'] ?? '');
        break;
      case 'needs_decision':
        this.pending.set(ev['match']);
        break;
      case 'done':
        this.pushUp('store', 'good', `Done — book #${ev['book_id']} "${ev['title']}" (${ev['chunks']} chunks, ${ev['kind']}).`);
        this.upDone.set(ev);
        this.file.set(null);
        this.fileName.set('');
        break;
      case 'error':
        this.pushUp('error', 'error', ev['error'] ?? 'unknown error');
        break;
    }
  }

  private pushUp(stage: string, level: string, text: string): void {
    this.upLogs.update((ls) => [...ls, { stage, level, text }]);
  }

  // ---------------- browse + search ----------------
  async openDetail(id: number): Promise<void> {
    try {
      this.openBook.set(await this.api.detail(id));
    } catch {
      this.openBook.set(null);
    }
  }

  closeDetail(): void {
    this.openBook.set(null);
  }

  async runSearch(): Promise<void> {
    const q = this.searchQ().trim();
    if (!q) return;
    this.searching.set(true);
    try {
      this.searchHits.set(await this.api.search(q, this.fLang() || null, this.fSubject() || null));
    } catch {
      this.searchHits.set([]);
    } finally {
      this.searching.set(false);
    }
  }
}
