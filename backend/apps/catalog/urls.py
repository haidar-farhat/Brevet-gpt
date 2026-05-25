"""Manage Materials API routes (upload / browse / freeze / search)."""
from __future__ import annotations

from django.urls import path

from apps.catalog import views

urlpatterns = [
    path("api/materials", views.materials_list_view, name="materials_list"),
    path("api/materials/check", views.materials_check_view, name="materials_check"),
    path("api/materials/upload", views.materials_upload_view, name="materials_upload"),
    path("api/materials/search", views.materials_search_view, name="materials_search"),
    # Integer converter ensures the static paths above are matched first.
    path("api/materials/<int:book_id>", views.materials_detail_view, name="materials_detail"),
    path("api/materials/<int:book_id>/<str:action>", views.materials_status_view, name="materials_status"),
]
