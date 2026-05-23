import { Injectable } from '@angular/core';

/** One event from the /api/ask/stream SSE feed. */
export interface BrevetEvent {
  type: 'log' | 'token' | 'result' | 'error';
  [key: string]: any;
}

export interface AskHandlers {
  onEvent: (event: BrevetEvent) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}

/**
 * Talks to the Django async API. `ask()` POSTs to the SSE streaming endpoint and
 * parses `data: {...}\n\n` frames as they arrive (fetch + ReadableStream), so the
 * UI can render the AI's logs and answer tokens live.
 */
@Injectable({ providedIn: 'root' })
export class BrevetService {
  // The Django server (run: `python manage.py brevet`).
  private readonly base = 'http://localhost:8000';

  async ask(
    question: string,
    language: string | null,
    subject: string | null,
    handlers: AskHandlers,
    signal?: AbortSignal,
  ): Promise<void> {
    const body: Record<string, unknown> = { question };
    if (language) body['language'] = language;
    if (subject) body['subject'] = subject;

    let response: Response;
    try {
      response = await fetch(`${this.base}/api/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
      });
    } catch (e: any) {
      handlers.onError?.(`Cannot reach API at ${this.base}. Is the server running? (${e?.message ?? e})`);
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
              handlers.onEvent(JSON.parse(frame.slice(5).trim()) as BrevetEvent);
            } catch {
              /* ignore malformed frame */
            }
          }
        }
      }
    } catch (e: any) {
      handlers.onError?.(`Stream interrupted: ${e?.message ?? e}`);
    } finally {
      handlers.onDone?.();
    }
  }
}
