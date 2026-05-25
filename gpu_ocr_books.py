import os
import re
import sys
import json
import shutil
import time
import argparse
import unicodedata
import urllib.request
from statistics import median

import fitz
import cv2
import numpy as np
from tqdm import tqdm

import pytesseract
from pytesseract import Output
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont

# =========================
# CONFIG
# =========================

# Derive from this file's own location (the repo root) so paths are portable
# across machines instead of being pinned to one user's disk. The Django wrapper
# (apps/catalog/services/ocr.py) further overrides RESULTS_FOLDER to settings.RESULTS_DIR.
BOOKS_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(BOOKS_FOLDER, "results")

# Each input subfolder maps to: (tesseract lang, results subdir, ISO 639-1 code).
# The folder a book lives in decides its OCR language AND its routing metadata.
LANG_FOLDERS = {
    "english": ("eng", "eng", "en"),
    "french": ("fra", "fr", "fr"),
    "arabic": ("ara", "ar", "ar"),
}

# Arabic / RTL script ranges. Lines containing these are right-aligned and Unicode-
# normalised (NFKC) so the embedded *text layer* stays clean, logical Arabic — which
# is what the RAG pipeline extracts and embeds. We deliberately do NOT reshape glyphs
# into the PDF: reshaping would corrupt the extractable text the AI depends on.
_RTL_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def is_rtl(text):
    return bool(_RTL_RE.search(text or ""))


# Optional Arabic shaping for the *visual* PDF only (cursive joining + RTL order).
# The embedded/searchable text comes from the logical OCR sidecar, NOT from re-
# extracting this PDF, so shaping here can never corrupt what the RAG pipeline reads.
try:
    import arabic_reshaper as _arabic_reshaper
    from bidi.algorithm import get_display as _get_display
    _RTL_SHAPE = True
except Exception:  # pragma: no cover - libs optional
    _RTL_SHAPE = False


def shape_rtl(text):
    """Reshape + bidi-reorder Arabic for correct visual display in the rendered PDF."""
    if _RTL_SHAPE:
        try:
            return _get_display(_arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

# Where we drop language packs we have to fetch ourselves (no admin rights
# needed, unlike writing into the Tesseract install dir).
LOCAL_TESSDATA = os.path.join(BOOKS_FOLDER, "tessdata")
# "tessdata_fast" = good accuracy, smaller/faster. Switch to "tessdata_best"
# for maximum accuracy on accented French at the cost of speed.
TESSDATA_VARIANT = "tessdata_fast"
# Per-language override: Arabic recognition is markedly better with the 'best'
# model, so 'ara' always gets a local best copy (other languages stay fast).
TESSDATA_VARIANT_BY_LANG = {"ara": "tessdata_best"}


def _variant_for(lang):
    return TESSDATA_VARIANT_BY_LANG.get(lang, TESSDATA_VARIANT)


def _local_traineddata(lang):
    return os.path.join(LOCAL_TESSDATA, f"{lang}.traineddata")

# Render quality. NOTE: the previous version did `min(DPI/72, 2.0)`, which
# silently capped every page at ~144 DPI. Tesseract wants ~300 DPI, so we now
# render at the real DPI and only fall back if a page would exceed MAX_SIDE.
DPI = 300
MAX_SIDE = 4000

PAGE_SIZE = letter
MARGIN = 54
# Default to a Unicode TrueType font so accented French and math/science
# symbols (√, ≈, ², Greek) embed correctly. Falls back to Helvetica
# (Latin-1 only) if the system font is missing. See register_fonts().
FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
BODY_SIZE = 10
BODY_LEAD = 12.5
PARA_GAP = 6

# Heading sizes per level (used for both visual rendering and outline depth).
HEADING_SIZES = {1: 16, 2: 13, 3: 11}
HEADING_GAP = 6

# A line is a heading candidate only if it is much taller than body text,
# short, confidently recognised, and not sitting in a running-header margin.
HEADING_RATIOS = ((1.70, 1), (1.35, 2), (1.18, 3))  # (min size ratio, level)
HEADING_MAX_WORDS = 10
HEADING_MIN_CONF = 50
MARGIN_BAND = 0.08  # top/bottom 8% of the page = headers/footers, not content

# --psm 1 = automatic page segmentation with orientation/script detection
#           (handles single column AND multi-column textbook layouts).
TESSERACT_CONFIG = "--psm 1 -c preserve_interword_spaces=1"

# Subject inferred from filename/title -> routing & classification metadata.
SUBJECT_RULES = (
    (("mathemat", "math", "construire les math"), "Mathematics"),
    (("chimie", "chemis", "chem"), "Chemistry"),
    (("physiqu", "physic", "phys"), "Physics"),
    (("vie et de la terre", "biolog", "life and ea", "bio"), "Biology / Earth & Life Science"),
    (("grammaire", "grammar"), "Grammar"),
    (("lecture", "activites", "reading"), "Reading & Activities"),
    (("informat", "computer"), "Computer Science / Informatics"),
    (("workbook",), "Language Workbook"),
    (("francais", "français", "french"), "French Language"),
    (("english", "anglais"), "English Language"),
)

# =========================
# TESSERACT LOCATION
# =========================

def _locate_tesseract():
    for path in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ):
        if os.path.isfile(path):
            return path
    return shutil.which("tesseract")

_tess_path = _locate_tesseract()
if _tess_path:
    pytesseract.pytesseract.tesseract_cmd = _tess_path

try:
    _tess_version = pytesseract.get_tesseract_version()
except Exception:
    raise SystemExit(
        "\nERROR: Tesseract binary not found.\n"
        "  Install from https://github.com/UB-Mannheim/tesseract/wiki\n"
        "  Default install path expected: C:\\Program Files\\Tesseract-OCR\\\n"
    )

print(f"\nTesseract {_tess_version}: {_tess_path or 'PATH'}")

# =========================
# LANGUAGE PACKS
# =========================

def _install_tessdata_dir():
    if _tess_path:
        cand = os.path.join(os.path.dirname(_tess_path), "tessdata")
        if os.path.isdir(cand):
            return cand
    return None

def _download_traineddata(lang, dest):
    variant = _variant_for(lang)
    url = f"https://github.com/tesseract-ocr/{variant}/raw/main/{lang}.traineddata"
    print(f"  downloading {lang}.traineddata from {variant} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "ocr-books/1.0"})
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"could not download '{lang}': {e}\n"
            f"  Manually place {lang}.traineddata in {LOCAL_TESSDATA}\n"
            f"  (get it from https://github.com/tesseract-ocr/{_variant_for(lang)})"
        )

def ensure_languages(needed):
    """Make sure every needed language (plus 'osd' for --psm 1) is usable.

    Returns (tessdata_arg, available_set). If everything is already installed
    we use the system tessdata and tessdata_arg is "". Otherwise we assemble a
    self-contained local tessdata dir (copying what the install already has,
    downloading the rest) and point Tesseract at it via TESSDATA_PREFIX.
    """
    needed = set(needed) | {"osd"}
    try:
        installed = set(pytesseract.get_languages())
    except Exception:
        installed = set()

    # Languages pinned to a specific variant (Arabic -> best): never accept a
    # system/fast copy — require a local best copy, downloading it if missing.
    force = {l for l in needed if l in TESSDATA_VARIANT_BY_LANG and not os.path.isfile(_local_traineddata(l))}
    if needed <= installed and not force:
        return "", installed

    print(f"\nPreparing {LOCAL_TESSDATA} (missing {sorted(needed - installed)}; best: {sorted(force)})")
    os.makedirs(LOCAL_TESSDATA, exist_ok=True)
    install_dir = _install_tessdata_dir()
    available = set()

    for lang in sorted(needed):
        dest = _local_traineddata(lang)
        if os.path.isfile(dest):
            available.add(lang)
            continue
        # Forced-variant langs skip the install copy and download the right variant.
        src = None if lang in TESSDATA_VARIANT_BY_LANG else (
            os.path.join(install_dir, f"{lang}.traineddata") if install_dir else None)
        if src and os.path.isfile(src):
            shutil.copy2(src, dest)
            print(f"  copied {lang}.traineddata from install")
            available.add(lang)
            continue
        try:
            _download_traineddata(lang, dest)
            available.add(lang)
        except RuntimeError as e:
            print(f"  WARNING: {e}")
            # Fall back to the installed copy (e.g. fast Arabic) if download failed.
            fb = os.path.join(install_dir, f"{lang}.traineddata") if install_dir else None
            if fb and os.path.isfile(fb):
                shutil.copy2(fb, dest)
                available.add(lang)
                print(f"  fell back to installed {lang}.traineddata")

    # Env var is more robust than --tessdata-dir: pytesseract splits the config
    # string on whitespace without honouring quotes, which mangles paths.
    os.environ["TESSDATA_PREFIX"] = LOCAL_TESSDATA
    return "", available

# =========================
# METADATA HELPERS
# =========================

def register_fonts():
    """Use a Unicode TTF if available so OCR'd accents/symbols don't get dropped."""
    global FONT_NAME, FONT_BOLD
    pairs = (
        ("BookSans", r"C:\Windows\Fonts\arial.ttf", "BookSans-Bold", r"C:\Windows\Fonts\arialbd.ttf"),
        ("BookSans", r"C:\Windows\Fonts\DejaVuSans.ttf", "BookSans-Bold", r"C:\Windows\Fonts\DejaVuSans-Bold.ttf"),
    )
    for reg, reg_path, bold, bold_path in pairs:
        if os.path.isfile(reg_path) and os.path.isfile(bold_path):
            try:
                pdfmetrics.registerFont(TTFont(reg, reg_path))
                pdfmetrics.registerFont(TTFont(bold, bold_path))
                FONT_NAME, FONT_BOLD = reg, bold
                print(f"Font: {os.path.basename(reg_path)} (full Unicode)")
                return
            except Exception:
                continue
    print("Font: Helvetica (Latin-1 only; some symbols may not render)")

def prettify_stem(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.replace("_", " ").strip()
    return stem.title() if stem.islower() else stem

def clean_title(meta_title, filename):
    t = (meta_title or "").strip()
    t = re.sub(r"(?i)\.pdf$", "", t).strip()
    t = re.sub(r"(?i)\s*[-–]\s*pages?\s*\d+\s*[-–]\s*\d+\s*$", "", t).strip()
    t = t.strip(" -–")
    # U+FFFD is a true decode failure; trust the filename instead.
    if not t or "�" in t:
        t = prettify_stem(filename)
    return t

def subject_for(title, filename):
    hay = f"{title} {filename}".lower()
    for needles, subject in SUBJECT_RULES:
        if any(n in hay for n in needles):
            return subject
    return prettify_stem(filename)

# =========================
# RENDER + OCR
# =========================

def render_gray(page):
    zoom = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    if max(pix.width, pix.height) > MAX_SIDE:
        scale = (DPI / 72.0) * MAX_SIDE / max(pix.width, pix.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        # fitz emits RGB (not BGR); use the matching conversion.
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        img = img[:, :, 0]
    return np.ascontiguousarray(img)

def ocr_lines(img, lang, tessdata_arg):
    """OCR one page and return per-line records: text, height, conf, vertical span."""
    cfg = f"{TESSERACT_CONFIG} {tessdata_arg}".strip()
    data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=Output.DICT)

    grouped = {}
    order = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in grouped:
            grouped[key] = {"words": [], "heights": [], "confs": [], "tops": [], "bottoms": []}
            order.append(key)
        g = grouped[key]
        g["words"].append(txt)
        g["heights"].append(data["height"][i])
        g["confs"].append(conf)
        g["tops"].append(data["top"][i])
        g["bottoms"].append(data["top"][i] + data["height"][i])

    lines = []
    for key in order:
        g = grouped[key]
        lines.append({
            "block": key[0],
            "par": key[1],
            "text": " ".join(g["words"]),
            "height": float(median(g["heights"])),
            "conf": sum(g["confs"]) / len(g["confs"]),
            "top": min(g["tops"]),
            "bottom": max(g["bottoms"]),
            "words": len(g["words"]),
        })
    return lines

def detect_printed_page(lines, page_h):
    """Bare integer sitting in the top/bottom margin == the printed page number."""
    top_band = page_h * MARGIN_BAND
    bot_band = page_h * (1 - MARGIN_BAND)
    bottom_hit = top_hit = None
    for ln in lines:
        m = re.fullmatch(r"\d{1,4}", ln["text"].strip())
        if not m:
            continue
        if ln["bottom"] >= bot_band:
            bottom_hit = int(m.group())
        elif ln["top"] <= top_band:
            top_hit = int(m.group())
    return bottom_hit if bottom_hit is not None else top_hit

def join_paragraph(line_texts):
    out = ""
    for t in line_texts:
        t = t.strip()
        if not t:
            continue
        if not out:
            out = t
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t  # de-hyphenate a word split across a line break
        else:
            out = out + " " + t
    return re.sub(r"\s+", " ", out).strip()

def build_paragraphs(lines, page_h):
    """Collapse Tesseract lines into paragraphs (its block/par grouping)."""
    grouped = {}
    order = []
    for ln in lines:
        key = (ln["block"], ln["par"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(ln)

    paras = []
    for key in order:
        members = grouped[key]
        text = join_paragraph([m["text"] for m in members])
        if not text:
            continue
        paras.append({
            "text": text,
            "height": float(median([m["height"] for m in members])),
            "conf": sum(m["conf"] for m in members) / len(members),
            "words": sum(m["words"] for m in members),
            "top": min(m["top"] for m in members),
            "in_margin": (min(m["top"] for m in members) <= page_h * MARGIN_BAND
                          or max(m["bottom"] for m in members) >= page_h * (1 - MARGIN_BAND)),
            "level": 0,
        })
    return paras

def looks_like_heading(text):
    """Reject body fragments, equations and list items that happen to be large.

    Real section headings read like short titles: they start with a capital or
    section number, are mostly letters, and don't contain math operators.
    """
    t = text.strip()
    if len(t) < 3:
        return False
    if not (t[0].isupper() or t[0].isdigit()):  # lowercase start = sentence continuation
        return False
    if re.match(r"^[A-Za-z]\)|^\(?\d+\)", t):  # "a)", "(1)" list markers
        return False
    if re.search(r"[=+×÷≤≥<>]|\b\d+\s*[-+/*]\s*\d+\b", t):  # equations/formulas
        return False
    letters = sum(c.isalpha() for c in t)
    return letters >= 3 and letters / len(t) >= 0.5

def classify_headings(pages, body_height):
    if body_height <= 0:
        return
    for pg in pages:
        for p in pg["paras"]:
            if p["in_margin"] or p["words"] > HEADING_MAX_WORDS or p["conf"] < HEADING_MIN_CONF:
                continue
            if not looks_like_heading(p["text"]):
                continue
            ratio = p["height"] / body_height
            for min_ratio, level in HEADING_RATIOS:
                if ratio >= min_ratio:
                    p["level"] = level
                    break

# =========================
# WRAP
# =========================

def wrap_text(text, max_width, font, size):
    words = text.split(" ")
    if not words:
        return []
    lines = []
    cur = words[0]
    for w in words[1:]:
        trial = f"{cur} {w}"
        if stringWidth(trial, font, size) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

# =========================
# PDF RENDER
# =========================

def render_pdf(pages, out_path, meta):
    page_w, page_h = PAGE_SIZE
    max_width = page_w - 2 * MARGIN

    c = canvas.Canvas(out_path, pagesize=PAGE_SIZE)
    c.setTitle(meta["title"])
    c.setSubject(meta["subject"])
    c.setAuthor(meta["author"])
    c.setKeywords(", ".join(meta["keywords"]))
    c.setCreator("gpu_ocr_books.py")
    if hasattr(c, "setLang"):
        c.setLang(meta["iso"])

    outline_state = {"prev": -1, "last_title": None}
    key_counter = {"n": 0}

    def add_outline(title, level):
        # reportlab forbids jumping more than one level deeper at a time.
        lvl = min(level - 1, outline_state["prev"] + 1)
        title = re.sub(r"\s+", " ", title).strip()[:120]
        if title == outline_state["last_title"]:
            return  # collapse repeated running headers
        key_counter["n"] += 1
        key = f"h{key_counter['n']}"
        c.bookmarkPage(key)
        try:
            c.addOutlineEntry(title, key, level=max(lvl, 0), closed=False)
            outline_state["prev"] = max(lvl, 0)
            outline_state["last_title"] = title
        except Exception:
            pass

    def footer(label):
        c.setFont(FONT_NAME, 8)
        c.setFillGray(0.5)
        c.drawCentredString(page_w / 2, MARGIN * 0.55, label)
        c.setFillGray(0)

    for pg in pages:
        label = pg["label"]
        y = page_h - MARGIN

        def new_page():
            footer(label)
            c.showPage()
            return page_h - MARGIN

        for p in pg["paras"]:
            if p["level"]:
                size = HEADING_SIZES[p["level"]]
                font, lead, gap = FONT_BOLD, size + 3, HEADING_GAP
            else:
                size, font, lead, gap = BODY_SIZE, FONT_NAME, BODY_LEAD, PARA_GAP

            # Arabic/RTL: NFKC-normalise (canonical, logical text -> clean extractable
            # layer for the RAG pipeline) and right-align. Logical order is kept (no
            # glyph reshaping) so PyMuPDF re-extracts proper Arabic for embedding.
            text = p["text"]
            rtl = is_rtl(text)
            if rtl:
                text = unicodedata.normalize("NFKC", text)

            wrapped = wrap_text(text, max_width, font, size)
            if not wrapped:
                continue

            # Keep a heading with at least its first line (avoid orphan headings).
            if p["level"] and y - lead < MARGIN and y < page_h - MARGIN:
                y = new_page()
            if p["level"]:
                add_outline(text, p["level"])

            c.setFont(font, size)
            for ln in wrapped:
                if y - lead < MARGIN:
                    y = new_page()
                    c.setFont(font, size)
                if rtl:
                    c.drawRightString(page_w - MARGIN, y, shape_rtl(ln))
                else:
                    c.drawString(MARGIN, y, ln)
                y -= lead
            y -= gap

        footer(label)
        c.showPage()

    c.save()

# =========================
# PDF PROCESS
# =========================

def process_pdf(pdf_path, lang, tessdata_arg, out_dir, iso, sample=0, start=0):
    name = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    lo = min(start, len(doc))
    hi = min(lo + sample, len(doc)) if sample else len(doc)

    raw_author = doc.metadata.get("author")
    title = clean_title(doc.metadata.get("title"), name)
    subject = subject_for(title, name)
    suffix = "_sample" if (sample or start) else ""
    out_path = os.path.join(RESULTS_FOLDER, out_dir, os.path.splitext(name)[0] + suffix + ".pdf")

    print(f"\nProcessing: {name}  ({lang}) -> {title} [{subject}]")
    t0 = time.time()

    pages = []
    all_heights = []
    confs = []
    for i in tqdm(range(lo, hi), desc=name, unit="page"):
        page = doc.load_page(i)
        img = render_gray(page)
        lines = ocr_lines(img, lang, tessdata_arg)
        all_heights.extend(ln["height"] for ln in lines)
        confs.extend(ln["conf"] for ln in lines)
        printed = detect_printed_page(lines, img.shape[0])
        paras = build_paragraphs(lines, img.shape[0])
        label = f"p. {printed}" if printed is not None else f"p. {i + 1}*"
        pages.append({"number": printed if printed is not None else (i + 1),
                      "label": label, "paras": paras})
    doc.close()

    body_height = median(all_heights) if all_heights else 0
    classify_headings(pages, body_height)
    headings = sum(1 for pg in pages for p in pg["paras"] if p["level"])
    mean_conf = sum(confs) / len(confs) if confs else 0.0

    meta = {
        "title": title,
        "subject": subject,
        "author": raw_author if raw_author not in (None, "Admin") else "",
        "iso": iso,
        "keywords": [iso, subject, "OCR", "scanned textbook"],
    }
    render_pdf(pages, out_path, meta)

    # Logical-text sidecar: the RAG pipeline embeds THIS clean logical text, never the
    # re-extracted PDF — a PDF reader's bidi mangles Arabic / mixed-RTL on extraction.
    sidecar = os.path.splitext(out_path)[0] + ".ocr.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump({"pages": [
                {"number": pg.get("number"),
                 "paras": [{"text": p.get("text", ""), "level": p.get("level", 0)} for p in pg["paras"]]}
                for pg in pages
            ]}, fh, ensure_ascii=False)
    except Exception as e:  # pragma: no cover
        print(f"  WARNING: could not write OCR sidecar: {e}")

    print(f"  saved: {out_path}")
    print(f"  pages: {hi - lo} | headings: {headings} | mean OCR conf: {mean_conf:.1f} | {time.time() - t0:.1f}s")
    if mean_conf and mean_conf < 70:
        print(f"  WARNING: low OCR confidence ({mean_conf:.1f}); check scan quality / language pack.")

# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser(description="OCR scanned textbooks into clean, structured PDFs.")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="process only N pages per book (fast validation)")
    ap.add_argument("--start", type=int, default=0, metavar="N",
                    help="start at page N (0-based); use with --sample to probe mid-book")
    ap.add_argument("--only", choices=sorted(LANG_FOLDERS), help="process only one language folder")
    args = ap.parse_args()

    register_fonts()
    folders = {k: v for k, v in LANG_FOLDERS.items() if not args.only or k == args.only}

    # Figure out which languages we actually need, then make sure they exist.
    jobs = {}
    needed_langs = set()
    for folder, (lang, out_dir, iso) in folders.items():
        src = os.path.join(BOOKS_FOLDER, folder)
        if not os.path.isdir(src):
            continue
        pdfs = sorted(
            os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(".pdf")
        )
        if pdfs:
            jobs[folder] = (lang, out_dir, iso, pdfs)
            needed_langs.add(lang)

    if not jobs:
        print("No PDFs found.")
        return

    tessdata_arg, available = ensure_languages(needed_langs)

    total = sum(len(v[3]) for v in jobs.values())
    print(f"\nFound {total} PDF(s) across {len(jobs)} folder(s)")

    for folder, (lang, out_dir, iso, pdfs) in jobs.items():
        if lang not in available or "osd" not in available:
            print(f"\nSKIPPING '{folder}': language '{lang}' unavailable. "
                  f"Add {lang}.traineddata to {LOCAL_TESSDATA} and re-run.")
            continue
        os.makedirs(os.path.join(RESULTS_FOLDER, out_dir), exist_ok=True)
        for pdf in pdfs:
            process_pdf(pdf, lang, tessdata_arg, out_dir, iso, sample=args.sample, start=args.start)

    print("\nDONE")

if __name__ == "__main__":
    main()
