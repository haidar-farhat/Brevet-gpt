"""Root URL configuration.

Only the admin is mounted for now — the REST API arrives in a later phase.
"""
from __future__ import annotations

from django.contrib import admin
from django.urls import path

urlpatterns = [
    path("admin/", admin.site.urls),
]
