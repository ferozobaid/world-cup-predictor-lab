# Agent Guardrails

This repository contains a standalone Python ML/data engineering pipeline for international football analytics.

Do not rebuild, redesign, or scaffold a Next.js frontend here. The frontend is intentionally outside this v1 scope.

Core rules:

- Keep all model training and Python inference inside `ml-pipeline/`.
- The frontend integration boundary is static JSON in `outputs/predictions/`.
- Never require the frontend to train models or run Python at runtime.
- Treat chronology as a hard invariant: every feature for a match must use only information available before that match.
- Prefer explicit, reproducible CLI commands over notebook-only workflows.
- Optional enrichment files must be optional. The baseline pipeline must work with only the historical match CSV.
- Do not add neural networks, LSTMs, or deep learning frameworks.
- Preserve schema metadata in exported JSON so frontend consumers can version against stable artifacts.

