"""Tests for the mock caption model backend."""

from __future__ import annotations

import pytest
from PIL import Image

from wastecap.models.mock import MockModel
from wastecap.models.registry import clear_cache, get_model
from wastecap.schema import Candidate


@pytest.fixture
def tiny_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(128, 64, 32))


# ---------------------------------------------------------------------------
# MockModel basic behaviour
# ---------------------------------------------------------------------------


def test_mock_generates_n_candidates(tiny_image: Image.Image) -> None:
    model = MockModel()
    candidates = model.generate(tiny_image, prompt="describe this waste", n=3)
    assert len(candidates) == 3
    for c in candidates:
        assert isinstance(c, Candidate)
        assert c.model == "mock"
        assert c.text


def test_mock_is_deterministic(tiny_image: Image.Image) -> None:
    model = MockModel()
    c1 = model.generate(tiny_image, prompt="describe", n=2)
    c2 = model.generate(tiny_image, prompt="describe", n=2)
    assert [c.text for c in c1] == [c.text for c in c2]


def test_mock_normal_mode_parse_succeeded(tiny_image: Image.Image) -> None:
    model = MockModel()
    candidates = model.generate(tiny_image, prompt="describe", n=1)
    assert not candidates[0].parse_failed
    assert candidates[0].raw_parsed is not None


def test_mock_parse_fail_mode(monkeypatch: pytest.MonkeyPatch, tiny_image: Image.Image) -> None:
    monkeypatch.setenv("WASTECAP_MOCK_MODE", "parse_fail")
    model = MockModel()
    candidates = model.generate(tiny_image, prompt="describe", n=2)
    for c in candidates:
        assert c.parse_failed is True


def test_mock_free_text_mode(monkeypatch: pytest.MonkeyPatch, tiny_image: Image.Image) -> None:
    monkeypatch.setenv("WASTECAP_MOCK_MODE", "free_text")
    model = MockModel()
    candidates = model.generate(tiny_image, prompt="describe", n=1)
    assert not candidates[0].parse_failed
    assert candidates[0].text  # non-empty string


def test_mock_prompt_id_stored(tiny_image: Image.Image) -> None:
    model = MockModel()
    candidates = model.generate(tiny_image, prompt="p", n=1, prompt_id="test_prompt_v1")
    assert candidates[0].prompt_id == "test_prompt_v1"


def test_mock_does_not_support_grounding() -> None:
    assert not MockModel().supports_grounding


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_returns_mock() -> None:
    clear_cache()
    model = get_model("mock")
    assert model.name == "mock"


def test_registry_caches_instance() -> None:
    clear_cache()
    m1 = get_model("mock")
    m2 = get_model("mock")
    assert m1 is m2


def test_registry_unknown_backend_raises() -> None:
    clear_cache()
    with pytest.raises(ValueError, match="Unknown backend"):
        get_model("unicorn_model")


def test_registry_env_var_default(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_cache()
    monkeypatch.setenv("WASTECAP_BACKEND", "mock")
    model = get_model()
    assert model.name == "mock"
