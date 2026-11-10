"""
Tests for wastecap.prompts — template loading and structured-output parsing.
All tests are offline (no model calls, no downloads).
"""

from __future__ import annotations

import json

import pytest

from wastecap.prompts import (
    ParseResult,
    RenderedPrompt,
    crop_bbox,
    get_prompt_id,
    parse_structured_output,
    reload_prompts,
    render_prompt,
)


@pytest.fixture(autouse=True)
def _clear_prompt_cache() -> None:
    reload_prompts()


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    def test_mock_scene_returns_rendered_prompt(self) -> None:
        rp = render_prompt("mock", "scene_level")
        assert isinstance(rp, RenderedPrompt)
        assert rp.prompt_id == "mock_scene_v1"
        assert len(rp.text) > 0

    def test_mock_object_substitutes_object_id(self) -> None:
        rp = render_prompt("mock", "object_level", object_id="o3")
        assert "o3" in rp.text
        assert rp.prompt_id == "mock_object_v1"

    def test_blip2_scene_has_no_system(self) -> None:
        rp = render_prompt("blip2", "scene_level")
        assert rp.system is None

    def test_qwen_scene_has_system_message(self) -> None:
        rp = render_prompt("qwen_vl", "scene_level")
        assert rp.system is not None
        assert len(rp.system) > 0

    def test_qwen_object_substitutes_object_id(self) -> None:
        rp = render_prompt("qwen_vl", "object_level", object_id="o7")
        assert "o7" in rp.text

    def test_blip2_object_substitutes_object_id(self) -> None:
        rp = render_prompt("blip2", "object_level", object_id="o2")
        assert "o2" in rp.text

    def test_object_level_without_object_id_raises(self) -> None:
        with pytest.raises(ValueError, match="object_id"):
            render_prompt("mock", "object_level")

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="level"):
            render_prompt("mock", "image_level")  # type: ignore[arg-type]

    def test_unknown_model_falls_back_to_default(self) -> None:
        # A model with no YAML falls back to default.yaml
        rp = render_prompt("nonexistent_model", "scene_level")
        assert len(rp.text) > 0

    def test_all_models_have_both_levels(self) -> None:
        for model in ("mock", "blip2", "qwen_vl", "llava"):
            scene = render_prompt(model, "scene_level")
            obj = render_prompt(model, "object_level", object_id="o1")
            assert scene.prompt_id
            assert obj.prompt_id

    def test_prompt_ids_are_unique_across_models(self) -> None:
        ids = set()
        for model in ("mock", "blip2", "qwen_vl", "llava"):
            for level in ("scene_level", "object_level"):
                kw = {"object_id": "o1"} if level == "object_level" else {}
                pid = render_prompt(model, level, **kw).prompt_id
                ids.add(pid)
        # All 8 (model, level) combinations should have distinct ids
        assert len(ids) == 8


# ---------------------------------------------------------------------------
# get_prompt_id
# ---------------------------------------------------------------------------


class TestGetPromptId:
    def test_returns_id_string(self) -> None:
        pid = get_prompt_id("mock", "scene_level")
        assert pid == "mock_scene_v1"

    def test_blip2_scene_id(self) -> None:
        assert get_prompt_id("blip2", "scene_level") == "blip2_scene_v1"

    def test_qwen_object_id(self) -> None:
        assert get_prompt_id("qwen_vl", "object_level") == "qwen_object_v1"


# ---------------------------------------------------------------------------
# parse_structured_output — Stage 1: direct JSON
# ---------------------------------------------------------------------------


class TestParseStage1Direct:
    def _scene_payload(self) -> dict:
        return {
            "objects": [
                {
                    "object_id": "o1",
                    "category": "bottle",
                    "material": "plastic",
                    "condition": "crushed",
                    "count": 1,
                    "location": "lower left",
                    "caption": "crushed plastic bottle",
                }
            ],
            "scene_context": "outdoor, cluttered",
        }

    def test_valid_json_parses_correctly(self) -> None:
        text = json.dumps(self._scene_payload())
        result = parse_structured_output(text)
        assert not result.parse_failed
        assert result.data["scene_context"] == "outdoor, cluttered"
        assert result.data["objects"][0]["category"] == "bottle"

    def test_parse_result_has_raw_parsed(self) -> None:
        text = json.dumps(self._scene_payload())
        result = parse_structured_output(text)
        assert result.raw_parsed is not None
        assert "objects" in result.raw_parsed

    def test_count_coerced_to_int(self) -> None:
        payload = self._scene_payload()
        payload["objects"][0]["count"] = "2"  # string, should be coerced
        result = parse_structured_output(json.dumps(payload))
        assert result.data["objects"][0]["count"] == 2

    def test_object_level_flat_dict(self) -> None:
        payload = {
            "object_id": "o1",
            "category": "can",
            "material": "metal",
            "condition": "deformed",
            "count": 1,
            "location": "centre",
            "caption": "deformed metal can",
        }
        result = parse_structured_output(json.dumps(payload))
        assert not result.parse_failed
        assert result.data["category"] == "can"
        assert result.data["material"] == "metal"

    def test_json_array_of_objects_is_wrapped(self) -> None:
        result = parse_structured_output(
            '[{"object_id":"o1","category":"bottle","material":"plastic","caption":"bottle"}]'
        )
        assert not result.parse_failed
        assert result.data["objects"][0]["category"] == "bottle"


# ---------------------------------------------------------------------------
# parse_structured_output — vocabulary coercion
# ---------------------------------------------------------------------------


class TestVocabCoercion:
    def test_unknown_category_coerced(self) -> None:
        payload = {
            "objects": [
                {
                    "object_id": "o1",
                    "category": "spaceship",  # not in vocabulary
                    "material": "plastic",
                    "condition": "intact",
                    "count": 1,
                    "caption": "spaceship",
                }
            ],
            "scene_context": "test",
        }
        result = parse_structured_output(json.dumps(payload))
        assert result.data["objects"][0]["category"] == "unknown"
        # Original preserved in _raw
        assert result.data["objects"][0]["_raw"]["category"] == "spaceship"

    def test_unknown_material_coerced(self) -> None:
        payload = {
            "objects": [
                {
                    "object_id": "o1",
                    "category": "bottle",
                    "material": "unobtanium",
                    "condition": "intact",
                    "count": 1,
                    "caption": "bottle",
                }
            ],
            "scene_context": "",
        }
        result = parse_structured_output(json.dumps(payload))
        assert result.data["objects"][0]["material"] == "unknown"
        assert result.data["objects"][0]["_raw"]["material"] == "unobtanium"

    def test_valid_vocab_not_coerced(self) -> None:
        payload = {
            "objects": [
                {
                    "object_id": "o1",
                    "category": "carton",
                    "material": "cardboard",
                    "condition": "torn",
                    "count": 1,
                    "caption": "torn cardboard carton",
                }
            ],
            "scene_context": "indoor",
        }
        result = parse_structured_output(json.dumps(payload))
        obj = result.data["objects"][0]
        assert obj["category"] == "carton"
        assert obj["material"] == "cardboard"
        assert obj["condition"] == "torn"
        assert "_raw" not in obj

    def test_case_insensitive_coercion(self) -> None:
        payload = {
            "objects": [
                {
                    "object_id": "o1",
                    "category": "Bottle",  # uppercase
                    "material": "PLASTIC",
                    "condition": "Crushed",
                    "count": 1,
                    "caption": "crushed plastic bottle",
                }
            ],
            "scene_context": "",
        }
        result = parse_structured_output(json.dumps(payload))
        obj = result.data["objects"][0]
        assert obj["category"] == "bottle"
        assert obj["material"] == "plastic"
        assert obj["condition"] == "crushed"


# ---------------------------------------------------------------------------
# parse_structured_output — Stage 2: regex block extraction
# ---------------------------------------------------------------------------


class TestParseStage2Regex:
    def test_json_wrapped_in_prose(self) -> None:
        payload = {"objects": [], "scene_context": "cluttered outdoor scene"}
        text = f"Here is my analysis:\n{json.dumps(payload)}\nThank you."
        result = parse_structured_output(text)
        assert not result.parse_failed
        assert result.data["scene_context"] == "cluttered outdoor scene"

    def test_json_with_leading_explanation(self) -> None:
        payload = {
            "category": "bottle",
            "material": "plastic",
            "condition": "intact",
            "count": 1,
            "object_id": "o1",
            "caption": "intact plastic bottle",
        }
        text = "The waste object is: " + json.dumps(payload)
        result = parse_structured_output(text)
        assert not result.parse_failed
        assert result.data["category"] == "bottle"


# ---------------------------------------------------------------------------
# parse_structured_output — Stage 3: code fence
# ---------------------------------------------------------------------------


class TestParseStage3Fence:
    def test_json_in_markdown_fence(self) -> None:
        payload = {
            "objects": [
                {
                    "object_id": "o1",
                    "category": "can",
                    "material": "metal",
                    "condition": "intact",
                    "count": 1,
                    "caption": "intact metal can",
                }
            ],
            "scene_context": "indoor",
        }
        text = f"```json\n{json.dumps(payload)}\n```"
        result = parse_structured_output(text)
        assert not result.parse_failed
        assert result.data["objects"][0]["category"] == "can"

    def test_json_in_plain_fence(self) -> None:
        payload = {"objects": [], "scene_context": "no waste visible"}
        text = f"```\n{json.dumps(payload)}\n```"
        result = parse_structured_output(text)
        assert not result.parse_failed

    def test_json_after_fence_thinking(self) -> None:
        payload = {
            "category": "wrapper",
            "material": "plastic",
            "condition": "torn",
            "count": 1,
            "object_id": "o1",
            "caption": "torn plastic wrapper",
        }
        text = "Let me think...\n```json\n" + json.dumps(payload) + "\n```\n"
        result = parse_structured_output(text)
        assert not result.parse_failed


# ---------------------------------------------------------------------------
# parse_structured_output — Fallback: free text
# ---------------------------------------------------------------------------


class TestParseFallback:
    def test_malformed_json_sets_parse_failed(self) -> None:
        result = parse_structured_output("{ bad json :")
        assert result.parse_failed is True
        assert result.raw_parsed is None

    def test_plain_text_sets_parse_failed(self) -> None:
        result = parse_structured_output("A crushed plastic bottle lies on the pavement.")
        assert result.parse_failed is True

    def test_parse_failed_preserves_text(self) -> None:
        text = "A crushed plastic bottle lies on the pavement."
        result = parse_structured_output(text)
        assert result.data["caption"] == text
        assert result.data.get("parse_error") is True

    def test_parse_failed_has_no_raw_parsed(self) -> None:
        result = parse_structured_output("garbage text")
        assert result.raw_parsed is None

    def test_fenced_json_is_parsed_before_prose(self) -> None:
        result = parse_structured_output(
            'Here is the result:\n```json\n{"caption":"plastic bottle"}\n```'
        )
        assert not result.parse_failed
        assert result.data["caption"] == "plastic bottle"

    def test_json_embedded_in_prose_is_parsed(self) -> None:
        result = parse_structured_output('Result: {"caption":"metal can"} Thanks.')
        assert not result.parse_failed
        assert result.data["caption"] == "metal can"

    def test_python_dict_output_is_parsed_safely(self) -> None:
        result = parse_structured_output("{'caption': 'paper wrapper', 'count': 1}")
        assert not result.parse_failed
        assert result.data["caption"] == "paper wrapper"

    def test_empty_string_falls_back(self) -> None:
        result = parse_structured_output("")
        assert result.parse_failed is True

    def test_none_objects_list_falls_back_gracefully(self) -> None:
        # Valid JSON but wrong shape — should not crash
        result = parse_structured_output('{"objects": null, "scene_context": "x"}')
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Qwen grounding box extraction
# ---------------------------------------------------------------------------


class TestGroundingBoxExtraction:
    def test_extracts_qwen_grounding_boxes(self) -> None:
        text = (
            '{"objects": [{"object_id": "o1", "category": "bottle", '
            '"material": "plastic", "condition": "intact", "count": 1, '
            '"caption": "intact plastic bottle"}], "scene_context": "test"}\n'
            "<ref>bottle</ref><box>[[100,200,300,400]]</box>"
        )
        result = parse_structured_output(text)
        assert result.grounding_boxes is not None
        assert len(result.grounding_boxes) == 1
        assert result.grounding_boxes[0] == [100.0, 200.0, 300.0, 400.0]

    def test_multiple_grounding_boxes(self) -> None:
        text = (
            '{"objects": [], "scene_context": ""}'
            "<ref>bottle</ref><box>[[10,20,30,40]]</box>"
            "<ref>can</ref><box>[[50,60,70,80]]</box>"
        )
        result = parse_structured_output(text)
        assert result.grounding_boxes is not None
        assert len(result.grounding_boxes) == 2

    def test_no_grounding_boxes_returns_none(self) -> None:
        text = '{"objects": [], "scene_context": "test"}'
        result = parse_structured_output(text)
        assert result.grounding_boxes is None

    def test_grounding_present_on_fallback_text(self) -> None:
        text = "plain text <ref>bottle</ref><box>[[10,20,30,40]]</box>"
        result = parse_structured_output(text)
        # Grounding tags are a valid structured model response.
        assert result.parse_failed is False
        assert result.grounding_boxes is not None
        assert result.data["objects"][0]["caption"] == "bottle"


# ---------------------------------------------------------------------------
# crop_bbox
# ---------------------------------------------------------------------------


class TestCropBbox:
    def test_crops_correct_region(self) -> None:
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (200, 200), color=(100, 150, 200))
        crop = crop_bbox(img, [50.0, 50.0, 60.0, 60.0])
        # Crop should be roughly bbox + padding, staying within image
        assert crop.width > 0
        assert crop.height > 0
        assert crop.width <= 200
        assert crop.height <= 200

    def test_crop_is_rgb(self) -> None:
        from PIL import Image as PILImage

        img = PILImage.new("RGBA", (100, 100), color=(100, 150, 200, 255))
        crop = crop_bbox(img, [10.0, 10.0, 50.0, 50.0])
        assert crop.mode == "RGB"

    def test_crop_clamps_to_image_bounds(self) -> None:
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (100, 100))
        # bbox near the edge — padding would go out of bounds
        crop = crop_bbox(img, [90.0, 90.0, 20.0, 20.0])
        assert crop.width <= 100
        assert crop.height <= 100
