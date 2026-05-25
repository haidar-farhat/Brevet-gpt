from __future__ import annotations

from django.apps import AppConfig


class RagConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.rag"
    label = "rag"
    verbose_name = "RAG"
