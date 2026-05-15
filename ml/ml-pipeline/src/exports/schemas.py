"""Pydantic schemas for exported frontend JSON."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExportMetadata(BaseModel):
    schema_version: str
    model_version: str
    generated_at: str
    artifact: str
    source: dict[str, Any] = Field(default_factory=dict)


class FrontendExport(BaseModel):
    metadata: ExportMetadata
    data: Any

