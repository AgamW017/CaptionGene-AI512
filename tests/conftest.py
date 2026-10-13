"""Shared pytest fixtures for WasteCap tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Point to test configs so taxonomy tests don't rely on cwd
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def set_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set environment variables for every test.

    - WASTECAP_BACKEND=mock  (never loads weights)
    - WASTECAP_DB_PATH=<tmp> (isolated DB per test)
    - WASTECAP_CONFIGS_DIR   (points to project configs/)
    - WASTECAP_REVIEWER      (deterministic reviewer name)
    """
    monkeypatch.setenv("WASTECAP_BACKEND", "mock")
    monkeypatch.setenv("WASTECAP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("WASTECAP_CONFIGS_DIR", str(CONFIGS_DIR))
    monkeypatch.setenv("WASTECAP_REVIEWER", "test_reviewer")
    monkeypatch.setenv("WASTECAP_LOG_LEVEL", "WARNING")

    # Clear taxonomy LRU cache so each test gets a fresh load with the patched env.
    from wastecap import taxonomy

    taxonomy.reload_taxonomy()

    # Clear model registry cache so backend is fresh per test.
    from wastecap.models import registry

    registry.clear_cache()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Initialised DB in a temporary directory."""
    from wastecap.store import init_db

    path = tmp_path / "test.db"
    return init_db(path)


@pytest.fixture
def fixture_annotations() -> list[dict]:
    """Load all fixture annotation JSONs as raw dicts."""
    ann_dir = FIXTURES_DIR / "annotations"
    return [json.loads(p.read_text()) for p in sorted(ann_dir.glob("*.json"))]


@pytest.fixture
def fixture_records(fixture_annotations: list[dict]) -> list:
    """Load all fixture annotations as validated AnnotationRecord objects."""
    from wastecap.schema import AnnotationRecord

    return [AnnotationRecord.model_validate(a) for a in fixture_annotations]
