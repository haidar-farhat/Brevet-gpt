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

# --- Applications --------------------------------------------------------
DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]
LOCAL_APPS = [
    "apps.catalog",
    "apps.rag",
]
INSTALLED_APPS = DJANGO_APPS + LOCAL_APPS

MIDDLEWARE = [
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
LLM_MAX_TOKENS = env.int("LLM_MAX_TOKENS", default=1024)
LLM_TIMEOUT = env.int("LLM_TIMEOUT", default=120)

# --- Retrieval -----------------------------------------------------------
RAG_CANDIDATES = env.int("RAG_CANDIDATES", default=20)            # per retriever
RAG_TOP_K = env.int("RAG_TOP_K", default=6)                       # chunks in final context
RAG_MAX_CONTEXT_TOKENS = env.int("RAG_MAX_CONTEXT_TOKENS", default=3000)
RAG_MIN_RELEVANCE = env.float("RAG_MIN_RELEVANCE", default=0.25)  # below => refuse (off-topic)
RAG_MAX_REFORMULATIONS = env.int("RAG_MAX_REFORMULATIONS", default=2)

# Chunking: token budget per chunk and overlap between adjacent chunks.
EMBED_CHUNK_TOKENS = env.int("EMBED_CHUNK_TOKENS", default=512)
EMBED_CHUNK_OVERLAP = env.int("EMBED_CHUNK_OVERLAP", default=64)
