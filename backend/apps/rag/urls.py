from __future__ import annotations

from django.urls import path

from apps.rag import views

urlpatterns = [
    path("api/ask", views.ask_view, name="ask"),
    path("api/health", views.health_view, name="health"),
]
