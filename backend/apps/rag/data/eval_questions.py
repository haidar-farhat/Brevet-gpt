"""Bilingual evaluation set grounded in the seeded corpus.

Mix of factual ("what is…") and problem-solving / rule-recall questions across
subjects and both languages. Extend freely — each item needs a question and
optionally a language/subject hint for deterministic routing.
"""
from __future__ import annotations

EVAL_QUESTIONS: list[dict] = [
    {"question": "How does the human eye form an image?", "language": "en", "subject": "physics"},
    {"question": "What is a mole in chemistry?", "language": "en", "subject": "chemistry"},
    {"question": "What is the role of a neuron?", "language": "en", "subject": "biology"},
    {"question": "What are the conditions for a quadrilateral to be a parallelogram?", "language": "en", "subject": "math"},
    {"question": "How do you sort text in a spreadsheet?", "language": "en", "subject": "informatics"},
    {"question": "Qu'est-ce qu'un neurone et quel est son rôle ?", "language": "fr", "subject": "biology"},
    {"question": "Qu'est-ce qu'une mole en chimie ?", "language": "fr", "subject": "chemistry"},
    {"question": "Comment l'œil forme-t-il une image ?", "language": "fr", "subject": "physics"},
    {"question": "Énonce le théorème de Pythagore.", "language": "fr", "subject": "math"},
    {"question": "Qu'est-ce qu'un complément d'objet direct ?", "language": "fr", "subject": "grammar"},
]
