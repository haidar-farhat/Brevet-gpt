"""Root URL configuration: Django admin + the RAG (ask) and catalog (materials) APIs."""
from __future__ import annotations

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("apps.rag.urls")),
    path("", include("apps.catalog.urls")),
]
