"""Settings shared by every environment.

Anything that is secret or varies per environment is read from the process
environment (12-factor); see ``.env.example`` for the complete list.
"""
from __future__ import annotations

from pathlib import Path

import environ

# backend/  (this file lives at backend/config/settings/base.py)
BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BASE_DIR.parent

env = environ.Env(
    DJANGO_DEBUG=(bool, False),
    DJANGO_ALLOWED_HOSTS=(list, ["localhost", "127.0.0.1"]),
)
environ.Env.read_env(BASE_DIR / ".env")

# --- Core ----------------------------------------------------------------
SECRET_KEY = env("DJANGO_SECRET_KEY")
DEBUG = env("DJANGO_DEBUG")
ALLOWED_HOSTS = env("DJANGO_ALLOWED_HOSTS")

# CORS — the Angular dev server origin(s) allowed to call the API.
CORS_ALLOWED_ORIGINS = env(
    "CORS_ALLOWED_ORIGINS",
    default=["http://localhost:4200", "http://127.0.0.1:4200"],
)

# --- Applications --------------------------------------------------------
DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]
THIRD_PARTY_APPS = [
    "corsheaders",
]
LOCAL_APPS = [
    "apps.catalog",
    "apps.rag",
]
INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",  # must precede CommonMiddleware
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# --- Database ------------------------------------------------------------
# utf8mb4 is mandatory: book titles carry French diacritics (é, ç, …).
DATABASES = {
    "default": {
        "ENGINE": env("DB_ENGINE", default="django.db.backends.mysql"),
        "NAME": env("DB_NAME"),
        "USER": env("DB_USER"),
        "PASSWORD": env("DB_PASSWORD", default=""),
        "HOST": env("DB_HOST", default="127.0.0.1"),
        "PORT": env("DB_PORT", default="3306"),
        "CONN_MAX_AGE": env.int("DB_CONN_MAX_AGE", default=60),
        "OPTIONS": {
            "charset": "utf8mb4",
            "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
        },
    }
}

# --- Password validation -------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# --- Internationalisation ------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = env("DJANGO_TIME_ZONE", default="UTC")
USE_I18N = True
USE_TZ = True

# --- Static files --------------------------------------------------------
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Domain-specific -----------------------------------------------------
# Root of the scanned-textbook corpus (contains english/ and french/).
ASSETS_DIR = Path(env("ASSETS_DIR", default=str(PROJECT_ROOT / "assets")))

# Clean, OCR'd structured PDFs (the embedding input; eng/ and fr/ subfolders).
RESULTS_DIR = Path(env("RESULTS_DIR", default=str(PROJECT_ROOT / "results")))

# --- Vector store --------------------------------------------------------
CHROMA_DIR = Path(env("CHROMA_DIR", default=str(PROJECT_ROOT / "chroma")))
CHROMA_COLLECTION = env("CHROMA_COLLECTION", default="brevet")

# --- Embeddings ----------------------------------------------------------
# "local" = sentence-transformers on CPU (offline, default); "openai" = API.
EMBEDDING_BACKEND = env("EMBEDDING_BACKEND", default="local")
LOCAL_EMBED_MODEL = env("LOCAL_EMBED_MODEL", default="BAAI/bge-m3")

# Only used when EMBEDDING_BACKEND=openai.
OPENAI_API_KEY = env("OPENAI_API_KEY", default="")
OPENAI_EMBED_MODEL = env("OPENAI_EMBED_MODEL", default="text-embedding-3-large")

# --- Generation (LM Studio, OpenAI-compatible local server) --------------
LMSTUDIO_BASE_URL = env("LMSTUDIO_BASE_URL", default="http://localhost:1234/v1")
LMSTUDIO_API_KEY = env("LMSTUDIO_API_KEY", default="lm-studio")
LMSTUDIO_MODEL = env("LMSTUDIO_MODEL", default="")  # blank => auto-detect loaded model
LLM_TEMPERATURE = env.float("LLM_TEMPERATURE", default=0.2)
LLM_MAX_TOKENS = env.int("LLM_MAX_TOKENS", default=1536)  # room for fuller, tutor-style answers
LLM_TIMEOUT = env.int("LLM_TIMEOUT", default=120)
# Reasoning models (e.g. Qwen3) otherwise spend the whole token budget on hidden
# chain-of-thought and emit EMPTY content. /no_think disables it (ignored by
# non-reasoning models); our structured Problem/Method/Step/Result IS the working.
LLM_NO_THINK = env.bool("LLM_NO_THINK", default=True)
# Floor on completion tokens: a reasoning model (Qwen3) needs headroom to "think"
# AND still emit the visible answer — without it a small per-call cap is consumed
# entirely by hidden reasoning and the answer comes out empty. It is a CAP, not a
# target (the model still stops when done), so short calls aren't slowed. 0 = off.
LLM_MIN_COMPLETION_TOKENS = env.int("LLM_MIN_COMPLETION_TOKENS", default=0)

# --- Retrieval -----------------------------------------------------------
RAG_CANDIDATES = env.int("RAG_CANDIDATES", default=20)            # per retriever
RAG_TOP_K = env.int("RAG_TOP_K", default=6)                       # chunks in final context
RAG_MAX_CONTEXT_TOKENS = env.int("RAG_MAX_CONTEXT_TOKENS", default=3000)
RAG_MIN_RELEVANCE = env.float("RAG_MIN_RELEVANCE", default=0.25)  # below => refuse (off-topic)
RAG_MAX_REFORMULATIONS = env.int("RAG_MAX_REFORMULATIONS", default=2)

# Chunking: token budget per chunk and overlap between adjacent chunks.
EMBED_CHUNK_TOKENS = env.int("EMBED_CHUNK_TOKENS", default=512)
EMBED_CHUNK_OVERLAP = env.int("EMBED_CHUNK_OVERLAP", default=64)

# --- Agentic RAG ---------------------------------------------------------
# Master switch. False => the original linear pipeline (byte-identical behaviour).
RAG_AGENTIC = env.bool("RAG_AGENTIC", default=True)

# Reranking (highest accuracy ROI, 0 LLM cost): cross-encoder reorders the
# fused candidates before the token-budget selection.
# "dense" = no extra model, reorders using the dense similarity (robust default).
# "cross_encoder" = best quality but loads a 2nd torch model — on Windows this can
# exceed the page-file/commit limit alongside bge-m3; raise the page file to use it.
RAG_RERANK = env.bool("RAG_RERANK", default=True)
RAG_RERANK_BACKEND = env("RAG_RERANK_BACKEND", default="dense")  # dense|cross_encoder|none
RAG_RERANK_MODEL = env("RAG_RERANK_MODEL", default="BAAI/bge-reranker-v2-m3")
RAG_RERANK_CANDIDATES = env.int("RAG_RERANK_CANDIDATES", default=12)
RAG_DENSE_SIM_WEIGHT = env.float("RAG_DENSE_SIM_WEIGHT", default=0.3)  # blend w/ cross score

# Context analysing (LLM grades relevance + sufficiency of retrieved chunks).
RAG_GRADE_CONTEXT = env.bool("RAG_GRADE_CONTEXT", default=True)
RAG_GRADE_MAX_CHUNKS = env.int("RAG_GRADE_MAX_CHUNKS", default=4)
RAG_SUFFICIENCY_MIN = env.float("RAG_SUFFICIENCY_MIN", default=0.5)

# Agentic refine loop (failure-aware query refinement when context is weak).
RAG_AGENT_MAX_LOOPS = env.int("RAG_AGENT_MAX_LOOPS", default=1)

# Reason-then-answer for problem-solving subjects.
RAG_REASON = env.bool("RAG_REASON", default=True)
RAG_REASON_SUBJECTS = env.list("RAG_REASON_SUBJECTS", default=["math", "physics", "chemistry"])
RAG_REASON_MAX_TOKENS = env.int("RAG_REASON_MAX_TOKENS", default=512)

# Self-verification of the answer's claims against the retrieved context.
RAG_VERIFY = env.bool("RAG_VERIFY", default=True)
RAG_VERIFY_MIN = env.float("RAG_VERIFY_MIN", default=0.5)
RAG_VERIFY_MAX_CLAIMS = env.int("RAG_VERIFY_MAX_CLAIMS", default=6)
RAG_VERIFY_ACTION = env("RAG_VERIFY_ACTION", default="warn")  # warn|revise|refuse

# Hard ceiling on LLM calls per answer (protects a slow CPU model from blowups).
RAG_AGENT_LLM_BUDGET = env.int("RAG_AGENT_LLM_BUDGET", default=8)

# Smart routing: refuse off-topic questions early (saves retrieval+generation),
# and ask a clarifying question when the request is too vague to answer.
RAG_SCOPE_GUARD = env.bool("RAG_SCOPE_GUARD", default=True)
RAG_CLARIFY = env.bool("RAG_CLARIFY", default=True)

# Routing breadth: when False (default), retrieval is NOT hard-filtered by the
# routed subject — a mis-route can't exclude the right chunks; reranking + LLM
# grading decide relevance across all subjects. Set True for stricter, faster
# single-subject search when routing is trusted.
RAG_SUBJECT_FILTER = env.bool("RAG_SUBJECT_FILTER", default=False)

# --- Decompose & solve (multi-part problems) -----------------------------
# When a routed math/physics/chemistry question is an exercise to SOLVE (apply
# textbook rules to a NEW problem), break it into self-contained sub-problems
# and solve each with a small, focused context. The small per-part prompt is the
# real fix for "no output" on a small local model (a long monolith prompt makes a
# 4B model stream zero tokens). Set RAG_SOLVE=False for the single-pass tutor prompt.
RAG_SOLVE = env.bool("RAG_SOLVE", default=True)
RAG_MAX_SUBPROBLEMS = env.int("RAG_MAX_SUBPROBLEMS", default=12)   # safety cap (covers a 9-part sheet)
RAG_SOLVE_TOP_K = env.int("RAG_SOLVE_TOP_K", default=3)            # chunks fed to each part
RAG_SOLVE_CONTEXT_TOKENS = env.int("RAG_SOLVE_CONTEXT_TOKENS", default=1400)
RAG_SOLVE_MAX_TOKENS = env.int("RAG_SOLVE_MAX_TOKENS", default=1000)  # room for a multi-task part
RAG_SOLVE_TEMPERATURE = env.float("RAG_SOLVE_TEMPERATURE", default=0.0)  # deterministic maths
# Retry a generation once with a leaner prompt when it streams zero tokens.
RAG_SOLVE_RETRY_EMPTY = env.bool("RAG_SOLVE_RETRY_EMPTY", default=True)
# Auto-continue an answer the model cut off at the token limit (finish_reason=length),
# instead of leaving it truncated mid-step. Bounded + budget-gated.
RAG_MAX_CONTINUATIONS = env.int("RAG_MAX_CONTINUATIONS", default=2)

# Never silently drop a long worksheet; only truncate beyond this hard cap.
RAG_MAX_QUESTION_CHARS = env.int("RAG_MAX_QUESTION_CHARS", default=6000)

# --- Semantic cache ------------------------------------------------------
# Serve near-duplicate questions from a cached answer (skips the whole pipeline).
RAG_CACHE = env.bool("RAG_CACHE", default=True)
RAG_CACHE_MIN_SIM = env.float("RAG_CACHE_MIN_SIM", default=0.97)  # high => only near-identical
RAG_CACHE_COLLECTION = env("RAG_CACHE_COLLECTION", default="query_cache")
