"""Production settings (stub — not used in the current phase).

Kept here so the security posture is explicit and version-controlled from day
one. Wire this module up via DJANGO_SETTINGS_MODULE=config.settings.production.
"""
from __future__ import annotations

from .base import *  # noqa: F401,F403

DEBUG = False

# HTTPS / transport hardening (enable once served behind TLS).
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 60 * 60 * 24 * 365
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
