from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_OPS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qdrant_ops.py"


def _ops():
    spec = importlib.util.spec_from_file_location("qdrant_ops", _OPS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recover_location_from_snapshot_name(monkeypatch):
    ops = _ops()
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    location = ops.snapshot_recover_location(
        "legal_documents",
        "legal_documents-123.snapshot",
    )
    assert location == (
        "http://127.0.0.1:6333/collections/legal_documents/snapshots/"
        "legal_documents-123.snapshot"
    )


def test_recover_location_from_absolute_file():
    ops = _ops()
    location = ops.snapshot_recover_location(
        "legal_documents",
        "/var/lib/qdrant/snapshots/legal_documents/snap.snapshot",
    )
    assert location == (
        "file:///var/lib/qdrant/snapshots/legal_documents/snap.snapshot"
    )


def test_recover_location_passes_through_http_and_file_uris():
    ops = _ops()
    http = "https://qdrant.internal:6333/collections/legal_documents/snapshots/a.snapshot"
    assert ops.snapshot_recover_location("legal_documents", http) == http
    file_uri = "file:///opt/backups/legal_documents.snapshot"
    assert ops.snapshot_recover_location("legal_documents", file_uri) == file_uri


def test_recover_location_rejects_empty():
    ops = _ops()
    with pytest.raises(ValueError):
        ops.snapshot_recover_location("legal_documents", "  ")
