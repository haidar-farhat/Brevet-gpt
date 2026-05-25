import { Injectable } from '@angular/core';

export interface MaterialBook {
  id: number;
  title: string;
  language: string;
  subject: string;
  school: string | null;
  grade: string | null;
  level: string;
  status: 'active' | 'frozen';
  total_pages: number | null;
  chunks: number;
  content_hash: string;
  replaces: number | null;
  processed_at: string | null;
}

/** Routing vocabularies for the dropdowns (from GET /api/taxonomy). */
export interface Taxonomy {
  languages: { code: string; name: string; native_name: string }[];
  schools: { code: string; name: string }[];
  grades: { code: string; name: string; ordinal: number }[];
  subjects: { code: string; name_en: string; name_fr: string }[];
}

export interface DupMatch {
  book_id: number;
  title: string;
  status: string;
  subject: string;
  language: string;
  chunks: number;
  match_reason: 'metadata' | 'content';
}

/** One event from the /api/materials/upload SSE feed. */
export interface UploadEvent {
  type: 'stage' | 'needs_decision' | 'done' | 'error';
  [key: string]: any;
}

export interface UploadHandlers {
  onEvent: (event: UploadEvent) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}

export interface UploadMeta {
  title: string;
  language: string;
  subject: string;
  school?: string | null;
  grade?: string | null;
  level: string;
  resolution?: string | null;
  target_id?: number | null;
}

/** Talks to the Django catalog (Manage Materials) API. */
@Injectable({ providedIn: 'root' })
export class MaterialsService {
  private readonly base = 'http://localhost:8000';

  async taxonomy(): Promise<Taxonomy> {
    const res = await fetch(`${this.base}/api/taxonomy`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async list(filters: Record<string, string | null> = {}): Promise<{ books: MaterialBook[]; ingest_in_progress: boolean }> {
    const qs = Object.entries(filters)
      .filter(([, v]) => v)
      .map(([k, v]) => `${k}=${encodeURIComponent(v as string)}`)
      .join('&');
    const res = await fetch(`${this.base}/api/materials${qs ? '?' + qs : ''}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async detail(id: number, offset = 0, limit = 50): Promise<any> {
    const res = await fetch(`${this.base}/api/materials/${id}?offset=${offset}&limit=${limit}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async check(meta: { title: string; language: string; subject: string }): Promise<DupMatch | null> {
    const res = await fetch(`${this.base}/api/materials/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(meta),
    });
    if (!res.ok) return null;
    return (await res.json()).match ?? null;
  }

  async setStatus(id: number, action: 'freeze' | 'unfreeze'): Promise<void> {
    const res = await fetch(`${this.base}/api/materials/${id}/${action}`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  }

  async remove(id: number): Promise<void> {
    const res = await fetch(`${this.base}/api/materials/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  }

  async search(q: string, language: string | null, subject: string | null): Promise<any[]> {
    const res = await fetch(`${this.base}/api/materials/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q, language, subject }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()).hits ?? [];
  }

  /** Upload a document (multipart) and stream OCR/embed progress as SSE. */
  async upload(file: File, meta: UploadMeta, handlers: UploadHandlers, signal?: AbortSignal): Promise<void> {
    const form = new FormData();
    form.append('file', file);
    form.append('title', meta.title);
    form.append('language', meta.language);
    form.append('subject', meta.subject);
    if (meta.school) form.append('school', meta.school);
    if (meta.grade) form.append('grade', meta.grade);
    form.append('level', meta.level || 'brevet');
    if (meta.resolution) form.append('resolution', meta.resolution);
    if (meta.target_id != null) form.append('target_id', String(meta.target_id));

    let response: Response;
    try {
      // No Content-Type header — the browser sets the multipart boundary.
      response = await fetch(`${this.base}/api/materials/upload`, { method: 'POST', body: form, signal });
    } catch (e: any) {
      if (e?.name !== 'AbortError') {
        handlers.onError?.(`Cannot reach API at ${this.base}. Is the server running? (${e?.message ?? e})`);
      }
      handlers.onDone?.();
      return;
    }

    if (!response.ok || !response.body) {
      handlers.onError?.(`HTTP ${response.status} ${response.statusText}`);
      handlers.onDone?.();
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep: number;
        while ((sep = buffer.indexOf('\n\n')) >= 0) {
          const frame = buffer.slice(0, sep).trim();
          buffer = buffer.slice(sep + 2);
          if (frame.startsWith('data:')) {
            try {
              handlers.onEvent(JSON.parse(frame.slice(5).trim()) as UploadEvent);
            } catch {
              /* ignore malformed frame */
            }
          }
        }
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') handlers.onError?.(`Upload interrupted: ${e?.message ?? e}`);
    } finally {
      handlers.onDone?.();
    }
  }
}
