"""Diagnostics for the quality/health of the embedded chunks (read-only).

    python manage.py chunk_health                      # full report
    python manage.py chunk_health --book-id 3
    python manage.py chunk_health --language fr --subject biology
    python manage.py chunk_health --samples 3          # also dump example chunks
    python manage.py chunk_health --query "le cycle de l'eau" --top 5

What to look for:
* token distribution clustered near EMBED_CHUNK_TOKENS (no spike of tiny or
  oversized chunks);
* high page coverage per book (low coverage => OCR produced little text);
* high heading attribution (chunks carry a section breadcrumb);
* near-zero OCR-noise / empty / U+FFFD / oversized flags;
* MySQL chunk count == Chroma vector count.
"""
from __future__ import annotations

import random
from collections import Counter

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser

from apps.catalog.models import Book, Chunk

TINY_TOKENS = 32          # below this a chunk is mostly a fragment
LARGE_TOKENS = 800        # well above target -> a page/sentence that didn't split
TRUNCATION_LIMIT = 8192   # bge-m3 / model context: chunks above this get cut off
MIN_ALNUM_RATIO = 0.50    # below this the text is likely OCR noise / symbols


def _pct(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isalnum() or ch.isspace() for ch in text) / len(text)


class Command(BaseCommand):
    help = "Report on the health/quality of the embedded chunks."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--book-id", type=int, default=None)
        parser.add_argument("--language", choices=("en", "fr"), default=None)
        parser.add_argument("--subject", default=None)
        parser.add_argument("--samples", type=int, default=0,
                            help="Dump N smallest, N largest and N random chunks.")
        parser.add_argument("--query", default=None,
                            help="Retrieval smoke test: embed this text and show top hits.")
        parser.add_argument("--top", type=int, default=5)

    def handle(self, *args: object, **options: object) -> None:
        qs = Chunk.objects.select_related("book")
        if options["book_id"]:
            qs = qs.filter(book_id=options["book_id"])
        if options["language"]:
            qs = qs.filter(language=options["language"])
        if options["subject"]:
            qs = qs.filter(subject__code=options["subject"])

        rows = list(qs.values(
            "id", "book_id", "book__title", "book__total_pages", "language",
            "subject__code", "page_start", "page_end", "heading_path",
            "token_count", "content", "content_hash", "vector_id",
        ))
        total = len(rows)
        if total == 0:
            self.stdout.write(self.style.WARNING("No chunks found for the given filters."))
            return

        self._section("OVERVIEW")
        tokens = [r["token_count"] or 0 for r in rows]
        self.stdout.write(f"  chunks                {total}")
        self.stdout.write(f"  books                 {len({r['book_id'] for r in rows})}")
        self.stdout.write(f"  tokens total          {sum(tokens):,}")
        self.stdout.write(
            f"  tokens/chunk          min {min(tokens)} | p5 {_pct(tokens, 5)} | "
            f"median {_pct(tokens, 50)} | mean {sum(tokens) // total} | "
            f"p95 {_pct(tokens, 95)} | max {max(tokens)}"
        )

        self._section("QUALITY FLAGS  (lower is better)")
        tiny = [r for r in rows if (r["token_count"] or 0) < TINY_TOKENS]
        large = [r for r in rows if (r["token_count"] or 0) > LARGE_TOKENS]
        truncated = [r for r in rows if (r["token_count"] or 0) > TRUNCATION_LIMIT]
        empty = [r for r in rows if not (r["content"] or "").strip()]
        noisy = [r for r in rows if _alnum_ratio(r["content"] or "") < MIN_ALNUM_RATIO]
        replacement = [r for r in rows if "�" in (r["content"] or "")]
        missing_vec = [r for r in rows if r["vector_id"] is None]

        hashes = Counter(r["content_hash"] for r in rows if r["content_hash"])
        dup_groups = {h: c for h, c in hashes.items() if c > 1}
        dup_chunks = sum(c - 1 for c in dup_groups.values())

        self._flag("tiny chunks (<%d tok)" % TINY_TOKENS, len(tiny), total)
        self._flag("large chunks (>%d tok)" % LARGE_TOKENS, len(large), total)
        self._flag("over model limit (>%d)" % TRUNCATION_LIMIT, len(truncated), total)
        self._flag("empty content", len(empty), total)
        self._flag("OCR noise (<%.0f%% alnum)" % (MIN_ALNUM_RATIO * 100), len(noisy), total)
        self._flag("contains U+FFFD", len(replacement), total)
        self._flag("duplicate content", dup_chunks, total, extra=f"{len(dup_groups)} groups")
        self._flag("missing vector_id", len(missing_vec), total)

        self._section("HEADING ATTRIBUTION  (hierarchical context)")
        with_heading = sum(1 for r in rows if (r["heading_path"] or "").strip())
        distinct_headings = len({r["heading_path"] for r in rows if (r["heading_path"] or "").strip()})
        self.stdout.write(f"  chunks with a breadcrumb   {with_heading}/{total} ({with_heading * 100 // total}%)")
        self.stdout.write(f"  distinct headings          {distinct_headings}")

        self._section("DISTRIBUTION")
        self._counts("language", Counter(r["language"] for r in rows))
        self._counts("subject", Counter(r["subject__code"] for r in rows))

        self._section("PER-BOOK")
        self.stdout.write(f"  {'book':45} {'chunks':>6} {'pages':>11} {'med.tok':>7} {'head%':>6}")
        for book_id in sorted({r["book_id"] for r in rows}):
            brows = [r for r in rows if r["book_id"] == book_id]
            title = (brows[0]["book__title"] or f"#{book_id}")[:44]
            total_pages = brows[0]["book__total_pages"] or 0
            covered = set()
            for r in brows:
                covered.update(range(r["page_start"], r["page_end"] + 1))
            cov = f"{len(covered)}/{total_pages}" if total_pages else f"{len(covered)}/?"
            med = _pct([r["token_count"] or 0 for r in brows], 50)
            head_pct = sum(1 for r in brows if (r["heading_path"] or "").strip()) * 100 // len(brows)
            low = total_pages and len(covered) < 0.6 * total_pages
            line = f"  {title:45} {len(brows):>6} {cov:>11} {med:>7} {head_pct:>5}%"
            self.stdout.write(self.style.WARNING(line + "  <- low page coverage") if low else line)

        self._chroma_consistency(options, total)

        if options["samples"]:
            self._dump_samples(rows, options["samples"])

        if options["query"]:
            self._query(options["query"], options["top"], options["language"], options["subject"])

    # ------------------------------------------------------------------ helpers
    def _section(self, title: str) -> None:
        self.stdout.write(self.style.MIGRATE_HEADING(f"\n== {title} =="))

    def _flag(self, label: str, count: int, total: int, extra: str = "") -> None:
        pct = count * 100 / total
        msg = f"  {label:30} {count:>6}  ({pct:4.1f}%)" + (f"  {extra}" if extra else "")
        self.stdout.write(self.style.ERROR(msg) if count else self.style.SUCCESS(msg))

    def _counts(self, label: str, counter: Counter) -> None:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
        self.stdout.write(f"  {label:10} {parts}")

    def _chroma_consistency(self, options: dict, mysql_total: int) -> None:
        self._section("STORE CONSISTENCY")
        try:
            from apps.catalog.services.vectorstore import get_collection

            collection = get_collection(settings.CHROMA_DIR, settings.CHROMA_COLLECTION)
            where = self._chroma_where(options)
            chroma_total = collection.count() if where is None else len(
                collection.get(where=where, include=[])["ids"]
            )
        except Exception as exc:  # pragma: no cover
            self.stdout.write(self.style.WARNING(f"  could not read Chroma: {exc}"))
            return
        ok = chroma_total == mysql_total
        line = f"  mysql={mysql_total}  chroma={chroma_total}"
        self.stdout.write(self.style.SUCCESS(line + "  (match)") if ok
                          else self.style.ERROR(line + "  <- MISMATCH"))

    @staticmethod
    def _chroma_where(options: dict):
        conds = []
        if options["language"]:
            conds.append({"language": options["language"]})
        if options["subject"]:
            conds.append({"subject": options["subject"]})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def _dump_samples(self, rows: list[dict], n: int) -> None:
        self._section(f"SAMPLES (smallest / largest / random {n})")
        by_tokens = sorted(rows, key=lambda r: r["token_count"] or 0)
        picked = by_tokens[:n] + by_tokens[-n:] + random.sample(rows, min(n, len(rows)))
        for r in picked:
            head = r["heading_path"] or "(no heading)"
            self.stdout.write(self.style.HTTP_INFO(
                f"\n  [{r['language']}/{r['subject__code']}] {r['book__title']} "
                f"p.{r['page_start']}-{r['page_end']} | {r['token_count']} tok | {head}"
            ))
            self.stdout.write(f"    {(r['content'] or '')[:240]}...")

    def _query(self, text: str, top: int, language: str | None, subject: str | None) -> None:
        self._section(f"RETRIEVAL SMOKE TEST  q={text!r}")
        from apps.catalog.services.embeddings import build_embedder
        from apps.catalog.services.vectorstore import get_collection

        vector = build_embedder().embed([text])[0]
        collection = get_collection(settings.CHROMA_DIR, settings.CHROMA_COLLECTION)
        where = self._chroma_where({"language": language, "subject": subject})
        res = collection.query(query_embeddings=[vector], n_results=top,
                               where=where, include=["metadatas", "documents", "distances"])
        for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
            self.stdout.write(self.style.HTTP_INFO(
                f"\n  sim={1 - dist:.3f}  [{meta['language']}/{meta['subject']}] "
                f"{meta['title']} p.{meta['page_start']}-{meta['page_end']} | "
                f"{meta.get('heading_path') or '(no heading)'}"
            ))
            self.stdout.write(f"    {doc[:200]}...")
