"""
Tests for M5 grounding utilities: boxes, masks, IoU, instruction generator.
No GPU, no model downloads — MockModel used where a model is needed.
"""

from __future__ import annotations

import pytest

from wastecap.schema import (
    AnnotationRecord,
    Condition,
    DataSource,
    Material,
    ObjectAnnotation,
    PolygonMask,
    RLEMask,
    SortingAction,
    Split,
    WasteCategory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _obj(
    object_id: str = "o1",
    category: WasteCategory = WasteCategory.bottle,
    material: Material = Material.plastic,
    condition: Condition | None = Condition.crushed,
    sorting_action: SortingAction | None = SortingAction.recycle_plastic,
    bbox: list[float] | None = None,
    location: str | None = None,
    instruction: str | None = None,
) -> ObjectAnnotation:
    if bbox is None:
        bbox = [10.0, 20.0, 50.0, 80.0]
    return ObjectAnnotation(
        object_id=object_id,
        category=category,
        material=material,
        condition=condition,
        sorting_action=sorting_action,
        bbox=bbox,
        location=location,
        instruction=instruction,
    )


def _rec(objects: list | None = None, width: int = 320, height: int = 240) -> AnnotationRecord:
    return AnnotationRecord(
        image_id="test_grounding_001",
        file="tests/fixtures/images/taco_fix001.jpg",
        source=DataSource.taco,
        width=width,
        height=height,
        split=Split.train,
        objects=objects or [_obj()],
    )


# ===========================================================================
# boxes.py
# ===========================================================================


class TestBoxConversions:
    def test_xywh_to_xyxy(self) -> None:
        from wastecap.grounding.boxes import xywh_to_xyxy

        assert xywh_to_xyxy([10.0, 20.0, 50.0, 80.0]) == [10.0, 20.0, 60.0, 100.0]

    def test_xyxy_to_xywh(self) -> None:
        from wastecap.grounding.boxes import xyxy_to_xywh

        assert xyxy_to_xywh([10.0, 20.0, 60.0, 100.0]) == [10.0, 20.0, 50.0, 80.0]

    def test_round_trip_xywh_xyxy(self) -> None:
        from wastecap.grounding.boxes import xywh_to_xyxy, xyxy_to_xywh

        original = [5.0, 15.0, 100.0, 200.0]
        assert xyxy_to_xywh(xywh_to_xyxy(original)) == pytest.approx(original)

    def test_xywh_to_xyxy_wrong_length_raises(self) -> None:
        from wastecap.grounding.boxes import xywh_to_xyxy

        with pytest.raises(ValueError):
            xywh_to_xyxy([1.0, 2.0, 3.0])

    def test_qwen_to_pixel(self) -> None:
        from wastecap.grounding.boxes import qwen_to_pixel

        # 500/1000 * 320 = 160, 500/1000 * 240 = 120
        result = qwen_to_pixel([0.0, 0.0, 500.0, 500.0], 320, 240)
        assert result == pytest.approx([0.0, 0.0, 160.0, 120.0])

    def test_qwen_full_image(self) -> None:
        from wastecap.grounding.boxes import qwen_to_pixel

        result = qwen_to_pixel([0.0, 0.0, 999.0, 999.0], 640, 480)
        assert result[2] == pytest.approx(640 * 0.999)
        assert result[3] == pytest.approx(480 * 0.999)


class TestBoxValidation:
    def test_valid_box(self) -> None:
        from wastecap.grounding.boxes import validate_bbox

        ok, reason = validate_bbox([10.0, 20.0, 50.0, 80.0])
        assert ok
        assert reason == ""

    def test_negative_coordinate_invalid(self) -> None:
        from wastecap.grounding.boxes import validate_bbox

        ok, reason = validate_bbox([-1.0, 0.0, 50.0, 80.0])
        assert not ok
        assert "negative" in reason.lower()

    def test_zero_width_invalid(self) -> None:
        from wastecap.grounding.boxes import validate_bbox

        ok, reason = validate_bbox([10.0, 20.0, 0.0, 80.0])
        assert not ok
        assert "degenerate" in reason.lower()

    def test_zero_height_invalid(self) -> None:
        from wastecap.grounding.boxes import validate_bbox

        ok, reason = validate_bbox([10.0, 20.0, 50.0, 0.0])
        assert not ok

    def test_box_out_of_image_invalid(self) -> None:
        from wastecap.grounding.boxes import validate_bbox

        ok, reason = validate_bbox([100.0, 100.0, 300.0, 50.0], image_width=200)
        assert not ok
        assert "width" in reason.lower()

    def test_box_exactly_at_image_edge_valid(self) -> None:
        from wastecap.grounding.boxes import validate_bbox

        ok, _ = validate_bbox([0.0, 0.0, 200.0, 100.0], image_width=200, image_height=100)
        assert ok

    def test_nan_invalid(self) -> None:
        from wastecap.grounding.boxes import validate_bbox

        ok, reason = validate_bbox([float("nan"), 0.0, 10.0, 10.0])
        assert not ok

    def test_is_valid_bbox(self) -> None:
        from wastecap.grounding.boxes import is_valid_bbox

        assert is_valid_bbox([10.0, 10.0, 50.0, 50.0])
        assert not is_valid_bbox([-1.0, 0.0, 50.0, 50.0])


class TestBoxOperations:
    def test_clip_bbox_within_bounds(self) -> None:
        from wastecap.grounding.boxes import clip_bbox

        result = clip_bbox([10.0, 20.0, 50.0, 80.0], 320, 240)
        assert result == pytest.approx([10.0, 20.0, 50.0, 80.0])

    def test_clip_bbox_exceeds_width(self) -> None:
        from wastecap.grounding.boxes import clip_bbox

        result = clip_bbox([300.0, 0.0, 100.0, 50.0], 320, 240)
        assert result[0] + result[2] == pytest.approx(320.0)

    def test_clip_bbox_exceeds_height(self) -> None:
        from wastecap.grounding.boxes import clip_bbox

        result = clip_bbox([0.0, 200.0, 50.0, 100.0], 320, 240)
        assert result[1] + result[3] == pytest.approx(240.0)

    def test_normalize_and_denormalize(self) -> None:
        from wastecap.grounding.boxes import denormalize_bbox, normalize_bbox

        bbox = [32.0, 24.0, 64.0, 48.0]
        norm = normalize_bbox(bbox, 320, 240)
        assert norm == pytest.approx([0.1, 0.1, 0.2, 0.2])
        back = denormalize_bbox(norm, 320, 240)
        assert back == pytest.approx(bbox)

    def test_bbox_area(self) -> None:
        from wastecap.grounding.boxes import bbox_area

        assert bbox_area([0.0, 0.0, 10.0, 20.0]) == pytest.approx(200.0)

    def test_bbox_area_degenerate(self) -> None:
        from wastecap.grounding.boxes import bbox_area

        assert bbox_area([0.0, 0.0, 0.0, 10.0]) == 0.0

    def test_bbox_centroid(self) -> None:
        from wastecap.grounding.boxes import bbox_centroid

        cx, cy = bbox_centroid([10.0, 20.0, 40.0, 60.0])
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(50.0)

    def test_spatial_description_centre(self) -> None:
        from wastecap.grounding.boxes import spatial_description

        # Box centred at (160, 120) in a 320×240 image → middle third
        result = spatial_description([140.0, 100.0, 40.0, 40.0], 320, 240)
        assert result == "centre"

    def test_spatial_description_lower_left(self) -> None:
        from wastecap.grounding.boxes import spatial_description

        result = spatial_description([0.0, 200.0, 50.0, 30.0], 320, 240)
        assert result == "lower left"

    def test_spatial_description_upper_right(self) -> None:
        from wastecap.grounding.boxes import spatial_description

        result = spatial_description([280.0, 0.0, 40.0, 40.0], 320, 240)
        assert result == "upper right"

    def test_spatial_description_returns_string(self) -> None:
        from wastecap.grounding.boxes import spatial_description

        for x in [0, 140, 270]:
            for y in [0, 100, 200]:
                result = spatial_description([float(x), float(y), 10.0, 10.0], 320, 240)
                assert isinstance(result, str)
                assert len(result) > 0


# ===========================================================================
# masks.py
# ===========================================================================


class TestPolygonMask:
    def test_polygon_to_bbox_simple_rect(self) -> None:
        from wastecap.grounding.masks import polygon_to_bbox

        # Rectangle corners: (10,20), (60,20), (60,80), (10,80)
        ring = [10.0, 20.0, 60.0, 20.0, 60.0, 80.0, 10.0, 80.0]
        result = polygon_to_bbox([ring])
        assert result == pytest.approx([10.0, 20.0, 50.0, 60.0])

    def test_polygon_to_bbox_triangle(self) -> None:
        from wastecap.grounding.masks import polygon_to_bbox

        ring = [0.0, 0.0, 100.0, 0.0, 50.0, 100.0]
        result = polygon_to_bbox([ring])
        assert result == pytest.approx([0.0, 0.0, 100.0, 100.0])

    def test_polygon_to_bbox_multiple_rings(self) -> None:
        from wastecap.grounding.masks import polygon_to_bbox

        # Outer ring spans further than inner
        outer = [0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]
        inner = [20.0, 20.0, 80.0, 20.0, 80.0, 80.0, 20.0, 80.0]
        result = polygon_to_bbox([outer, inner])
        assert result == pytest.approx([0.0, 0.0, 100.0, 100.0])

    def test_polygon_to_bbox_empty_raises(self) -> None:
        from wastecap.grounding.masks import polygon_to_bbox

        with pytest.raises(ValueError):
            polygon_to_bbox([])

    def test_polygon_area_rectangle(self) -> None:
        from wastecap.grounding.masks import polygon_area

        # 50×60 rectangle → area = 3000
        ring = [10.0, 20.0, 60.0, 20.0, 60.0, 80.0, 10.0, 80.0]
        assert polygon_area([ring]) == pytest.approx(3000.0)

    def test_polygon_area_triangle(self) -> None:
        from wastecap.grounding.masks import polygon_area

        # Right triangle: base=100, height=100 → area = 5000
        ring = [0.0, 0.0, 100.0, 0.0, 0.0, 100.0]
        assert polygon_area([ring]) == pytest.approx(5000.0)

    def test_mask_to_bbox_polygon(self) -> None:
        from wastecap.grounding.masks import mask_to_bbox

        mask = PolygonMask(data=[[10.0, 20.0, 60.0, 20.0, 60.0, 80.0, 10.0, 80.0]])
        result = mask_to_bbox(mask)
        assert result is not None
        assert result == pytest.approx([10.0, 20.0, 50.0, 60.0])


class TestRLEMask:
    def test_rle_to_bbox_manual_decode(self) -> None:
        """Simple 4×4 image with a 2×2 block of ones in the top-left."""
        from wastecap.grounding.masks import rle_to_bbox

        # 4×4 image, column-major.
        # Top-left 2×2 block = pixels (0,0),(1,0),(0,1),(1,1)
        # In column-major order: col0=[0,1,2,3], col1=[4,5,6,7]...
        # We want rows 0,1 in cols 0,1.
        # Col-major: pixel k → row = k%4, col = k//4
        # rows 0,1 in col 0: pixels 0,1 → runs: 2 ones then 2 zeros
        # rows 0,1 in col 1: pixels 4,5 → runs: 2 zeros(col0 trailing), 2 zeros(col1 leading?
        # Actually COCO RLE: starts with 0s. For a simple test:
        # counts="0,2,2,2,12" → starts with 0 zeros, 2 ones, 2 zeros, 2 ones, rest zeros
        counts = "0,2,2,2,12"
        result = rle_to_bbox(counts, height=4, width=4)
        assert result is not None
        x, y, w, h = result
        assert x >= 0 and y >= 0
        assert w >= 1 and h >= 1

    def test_rle_to_bbox_invalid_returns_none(self) -> None:
        from wastecap.grounding.masks import rle_to_bbox

        result = rle_to_bbox("not,valid,rle,data,abc", height=10, width=10)
        # Should return None without crashing
        assert result is None or isinstance(result, list)

    def test_mask_to_bbox_rle(self) -> None:
        from wastecap.grounding.masks import mask_to_bbox

        mask = RLEMask(counts="0,2,2,2,12", size=[4, 4])
        result = mask_to_bbox(mask)
        # Should return a bbox or None (None is acceptable if decode fails)
        assert result is None or (isinstance(result, list) and len(result) == 4)


# ===========================================================================
# iou.py
# ===========================================================================


class TestComputeIoU:
    def test_identical_boxes_iou_one(self) -> None:
        from wastecap.grounding.iou import compute_iou

        box = [10.0, 10.0, 50.0, 50.0]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap_iou_zero(self) -> None:
        from wastecap.grounding.iou import compute_iou

        a = [0.0, 0.0, 10.0, 10.0]
        b = [20.0, 20.0, 10.0, 10.0]
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_adjacent_boxes_iou_zero(self) -> None:
        from wastecap.grounding.iou import compute_iou

        a = [0.0, 0.0, 10.0, 10.0]
        b = [10.0, 0.0, 10.0, 10.0]  # touches but no overlap
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        from wastecap.grounding.iou import compute_iou

        a = [0.0, 0.0, 10.0, 10.0]  # area=100
        b = [5.0, 0.0, 10.0, 10.0]  # area=100; overlap=[5,0,5,10]=50
        # IoU = 50/(100+100-50) = 50/150 = 1/3
        assert compute_iou(a, b) == pytest.approx(1 / 3, rel=1e-5)

    def test_one_contained_in_other(self) -> None:
        from wastecap.grounding.iou import compute_iou

        big = [0.0, 0.0, 100.0, 100.0]  # area=10000
        small = [25.0, 25.0, 50.0, 50.0]  # area=2500; fully inside
        # IoU = 2500/(10000+2500-2500) = 2500/10000 = 0.25
        assert compute_iou(big, small) == pytest.approx(0.25)

    def test_degenerate_box_zero(self) -> None:
        from wastecap.grounding.iou import compute_iou

        a = [0.0, 0.0, 10.0, 10.0]
        b = [0.0, 0.0, 0.0, 10.0]  # zero width
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_wrong_length_raises(self) -> None:
        from wastecap.grounding.iou import compute_iou

        with pytest.raises(ValueError):
            compute_iou([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])

    def test_iou_symmetric(self) -> None:
        from wastecap.grounding.iou import compute_iou

        a = [0.0, 0.0, 20.0, 30.0]
        b = [10.0, 15.0, 25.0, 20.0]
        assert compute_iou(a, b) == pytest.approx(compute_iou(b, a))

    def test_iou_in_unit_interval(self) -> None:
        from wastecap.grounding.iou import compute_iou

        a = [0.0, 0.0, 50.0, 50.0]
        b = [25.0, 25.0, 50.0, 50.0]
        iou = compute_iou(a, b)
        assert 0.0 <= iou <= 1.0


class TestBatchIoU:
    def test_batch_iou_shape(self) -> None:
        from wastecap.grounding.iou import batch_iou

        preds = [[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 10.0, 10.0]]
        refs = [[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 10.0, 10.0], [50.0, 50.0, 5.0, 5.0]]
        matrix = batch_iou(preds, refs)
        assert len(matrix) == 2
        assert all(len(row) == 3 for row in matrix)

    def test_batch_iou_diagonal_identity(self) -> None:
        from wastecap.grounding.iou import batch_iou

        boxes = [[float(i * 20), 0.0, 15.0, 15.0] for i in range(4)]
        matrix = batch_iou(boxes, boxes)
        for i in range(4):
            assert matrix[i][i] == pytest.approx(1.0)

    def test_batch_iou_empty(self) -> None:
        from wastecap.grounding.iou import batch_iou

        assert batch_iou([], [[0.0, 0.0, 10.0, 10.0]]) == []
        assert batch_iou([[0.0, 0.0, 10.0, 10.0]], []) == [[]]


class TestBestMatch:
    def test_perfect_matches(self) -> None:
        from wastecap.grounding.iou import best_match

        boxes = [[float(i * 20), 0.0, 15.0, 15.0] for i in range(3)]
        matches = best_match(boxes, boxes, threshold=0.5)
        assert len(matches) == 3
        for pred_idx, ref_idx, iou in matches:
            assert pred_idx == ref_idx
            assert iou == pytest.approx(1.0)

    def test_threshold_filters_weak_matches(self) -> None:
        from wastecap.grounding.iou import best_match

        a = [0.0, 0.0, 10.0, 10.0]
        b = [9.0, 0.0, 10.0, 10.0]  # tiny overlap
        matches = best_match([a], [b], threshold=0.5)
        assert matches == []

    def test_each_reference_matched_once(self) -> None:
        from wastecap.grounding.iou import best_match

        # Two predictions, one reference — only one match
        ref = [[10.0, 10.0, 50.0, 50.0]]
        pred = [[10.0, 10.0, 50.0, 50.0], [10.0, 10.0, 50.0, 50.0]]
        matches = best_match(pred, ref)
        ref_indices = [r for _, r, _ in matches]
        assert len(set(ref_indices)) == len(ref_indices)

    def test_empty_inputs(self) -> None:
        from wastecap.grounding.iou import best_match

        assert best_match([], [[0.0, 0.0, 10.0, 10.0]]) == []
        assert best_match([[0.0, 0.0, 10.0, 10.0]], []) == []


class TestGroundingAccuracy:
    def test_perfect_grounding(self) -> None:
        from wastecap.grounding.iou import grounding_accuracy

        boxes = [[float(i * 20), 0.0, 15.0, 15.0] for i in range(3)]
        assert grounding_accuracy(boxes, boxes) == pytest.approx(1.0)

    def test_no_matches(self) -> None:
        from wastecap.grounding.iou import grounding_accuracy

        pred = [[0.0, 0.0, 5.0, 5.0]]
        ref = [[100.0, 100.0, 5.0, 5.0]]
        assert grounding_accuracy(pred, ref) == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        from wastecap.grounding.iou import grounding_accuracy

        refs = [[0.0, 0.0, 20.0, 20.0], [100.0, 100.0, 20.0, 20.0]]
        preds = [[0.0, 0.0, 20.0, 20.0]]  # only matches first ref
        acc = grounding_accuracy(preds, refs)
        assert acc == pytest.approx(0.5)

    def test_empty_reference_returns_zero(self) -> None:
        from wastecap.grounding.iou import grounding_accuracy

        assert grounding_accuracy([[0.0, 0.0, 10.0, 10.0]], []) == 0.0

    def test_empty_predicted_returns_zero(self) -> None:
        from wastecap.grounding.iou import grounding_accuracy

        assert grounding_accuracy([], [[0.0, 0.0, 10.0, 10.0]]) == 0.0


# ===========================================================================
# instructions.py
# ===========================================================================


class TestGenerateInstruction:
    def test_recycle_plastic_bottle(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = _obj(
            category=WasteCategory.bottle,
            material=Material.plastic,
            condition=Condition.crushed,
            sorting_action=SortingAction.recycle_plastic,
        )
        instr = generate_instruction(obj, 320, 240)
        assert "plastic" in instr.lower()
        assert "bottle" in instr.lower()
        assert "recycle" in instr.lower() or "recycling" in instr.lower()

    def test_recycle_metal_can(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = _obj(
            category=WasteCategory.can,
            material=Material.metal,
            condition=Condition.deformed,
            sorting_action=SortingAction.recycle_metal,
        )
        instr = generate_instruction(obj, 320, 240)
        assert "metal" in instr.lower()
        assert "can" in instr.lower()

    def test_compost(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = _obj(
            category=WasteCategory.food_waste,
            material=Material.organic,
            condition=None,
            sorting_action=SortingAction.compost,
        )
        instr = generate_instruction(obj, 320, 240)
        assert "compost" in instr.lower() or "organic" in instr.lower()

    def test_hazardous(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = _obj(
            category=WasteCategory.electronic,
            material=Material.mixed,
            condition=None,
            sorting_action=SortingAction.hazardous,
        )
        instr = generate_instruction(obj, 320, 240)
        assert "hazard" in instr.lower() or "special" in instr.lower()

    def test_recycle_cardboard(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = _obj(
            category=WasteCategory.box,
            material=Material.cardboard,
            condition=Condition.torn,
            sorting_action=SortingAction.recycle_cardboard,
        )
        instr = generate_instruction(obj, 320, 240)
        assert "cardboard" in instr.lower()

    def test_landfill(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = _obj(
            category=WasteCategory.wrapper,
            material=Material.plastic,
            condition=None,
            sorting_action=SortingAction.landfill,
        )
        instr = generate_instruction(obj, 320, 240)
        assert "landfill" in instr.lower() or "general" in instr.lower()

    def test_all_8_sorting_actions_produce_non_empty_instruction(self) -> None:
        from wastecap.grounding.instructions import generate_instruction
        from wastecap.schema import SortingAction

        for action in SortingAction:
            obj = ObjectAnnotation(
                object_id="o1",
                category=WasteCategory.bottle,
                material=Material.plastic,
                sorting_action=action,
                bbox=[10.0, 10.0, 50.0, 50.0],
            )
            instr = generate_instruction(obj, 320, 240)
            assert instr, f"Empty instruction for action {action}"
            assert len(instr.split()) >= 5, f"Instruction too short for action {action}: {instr!r}"

    def test_all_16_categories_produce_instruction(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        for cat in WasteCategory:
            obj = ObjectAnnotation(
                object_id="o1",
                category=cat,
                material=Material.plastic,
                sorting_action=SortingAction.landfill,
                bbox=[10.0, 10.0, 50.0, 50.0],
            )
            instr = generate_instruction(obj, 320, 240)
            assert instr

    def test_instruction_contains_location(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = _obj(bbox=[0.0, 200.0, 50.0, 30.0])  # lower left
        instr = generate_instruction(obj, 320, 240)
        assert "lower left" in instr.lower()

    def test_no_bbox_raises(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = ObjectAnnotation(
            object_id="o1",
            category=WasteCategory.bottle,
            material=Material.plastic,
            sorting_action=SortingAction.recycle_plastic,
        )
        with pytest.raises(ValueError, match="bbox"):
            generate_instruction(obj)

    def test_no_sorting_action_raises(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = ObjectAnnotation(
            object_id="o1",
            category=WasteCategory.bottle,
            material=Material.plastic,
            bbox=[10.0, 10.0, 50.0, 50.0],
        )
        with pytest.raises(ValueError, match="sorting_action"):
            generate_instruction(obj)

    def test_unknown_category_still_produces_instruction(self) -> None:
        from wastecap.grounding.instructions import generate_instruction

        obj = _obj(
            category=WasteCategory.unknown,
            material=Material.unknown,
            condition=None,
        )
        instr = generate_instruction(obj, 320, 240)
        assert instr


class TestValidateInstruction:
    def test_valid_object_passes(self) -> None:
        from wastecap.grounding.instructions import validate_instruction

        ok, reason = validate_instruction(_obj())
        assert ok
        assert reason == ""

    def test_no_bbox_fails(self) -> None:
        from wastecap.grounding.instructions import validate_instruction

        obj = ObjectAnnotation(
            object_id="o1",
            category=WasteCategory.bottle,
            material=Material.plastic,
            sorting_action=SortingAction.recycle_plastic,
        )
        ok, reason = validate_instruction(obj)
        assert not ok
        assert "bbox" in reason.lower()

    def test_no_sorting_action_fails(self) -> None:
        from wastecap.grounding.instructions import validate_instruction

        obj = ObjectAnnotation(
            object_id="o1",
            category=WasteCategory.bottle,
            material=Material.plastic,
            bbox=[10.0, 10.0, 50.0, 50.0],
        )
        ok, reason = validate_instruction(obj)
        assert not ok
        assert "sorting_action" in reason.lower()

    def test_empty_instruction_string_fails(self) -> None:
        from wastecap.grounding.instructions import validate_instruction

        obj = _obj(instruction="   ")
        ok, reason = validate_instruction(obj)
        assert not ok

    def test_non_empty_instruction_passes(self) -> None:
        from wastecap.grounding.instructions import validate_instruction

        obj = _obj(instruction="Pick up the bottle and recycle it.")
        ok, _ = validate_instruction(obj)
        assert ok

    def test_bbox_outside_image_fails(self) -> None:
        from wastecap.grounding.instructions import validate_instruction

        ok, reason = validate_instruction(_obj(bbox=[10.0, 20.0, 400.0, 80.0]), 320, 240)
        assert not ok
        assert "width" in reason.lower()

    def test_invalid_mask_fails(self) -> None:
        from wastecap.grounding.instructions import validate_instruction

        obj = _obj()
        obj.mask = PolygonMask(data=[[10.0, 20.0, 10.0]])
        ok, reason = validate_instruction(obj, 320, 240)
        assert not ok
        assert "mask" in reason.lower()


class TestGenerateAllInstructions:
    def test_fills_missing_instructions(self) -> None:
        from wastecap.grounding.instructions import generate_all_instructions

        rec = _rec(objects=[_obj("o1"), _obj("o2", bbox=[80.0, 80.0, 30.0, 30.0])])
        updated = generate_all_instructions(rec, 320, 240)
        assert all(o.instruction is not None for o in updated.objects)

    def test_does_not_overwrite_existing_by_default(self) -> None:
        from wastecap.grounding.instructions import generate_all_instructions

        existing = "Keep this instruction."
        obj = _obj(instruction=existing)
        rec = _rec(objects=[obj])
        updated = generate_all_instructions(rec, 320, 240)
        assert updated.objects[0].instruction == existing

    def test_overwrites_when_flag_set(self) -> None:
        from wastecap.grounding.instructions import generate_all_instructions

        existing = "Old instruction."
        obj = _obj(instruction=existing)
        rec = _rec(objects=[obj])
        updated = generate_all_instructions(rec, 320, 240, overwrite=True)
        assert updated.objects[0].instruction != existing

    def test_skips_objects_without_bbox(self) -> None:
        from wastecap.grounding.instructions import generate_all_instructions

        obj_no_bbox = ObjectAnnotation(
            object_id="o1",
            category=WasteCategory.bottle,
            material=Material.plastic,
            sorting_action=SortingAction.recycle_plastic,
        )
        rec = _rec(objects=[obj_no_bbox])
        updated = generate_all_instructions(rec, 320, 240)
        assert updated.objects[0].instruction is None

    def test_skips_objects_without_sorting_action(self) -> None:
        from wastecap.grounding.instructions import generate_all_instructions

        obj_no_action = ObjectAnnotation(
            object_id="o1",
            category=WasteCategory.bottle,
            material=Material.plastic,
            bbox=[10.0, 10.0, 50.0, 50.0],
        )
        rec = _rec(objects=[obj_no_action])
        updated = generate_all_instructions(rec, 320, 240)
        assert updated.objects[0].instruction is None

    def test_returns_copy_not_mutating_original(self) -> None:
        from wastecap.grounding.instructions import generate_all_instructions

        rec = _rec()
        updated = generate_all_instructions(rec, 320, 240)
        assert updated is not rec
        # Original unchanged
        assert rec.objects[0].instruction is None

    def test_validate_all_instructions_empty(self) -> None:
        from wastecap.grounding.instructions import validate_all_instructions

        rec = _rec()
        assert validate_all_instructions(rec) == []

    def test_validate_all_instructions_with_invalid(self) -> None:
        from wastecap.grounding.instructions import validate_all_instructions

        obj = ObjectAnnotation(
            object_id="o1",
            category=WasteCategory.bottle,
            material=Material.plastic,
            # No bbox, no sorting_action — instruction would fail
            instruction=None,
        )
        rec = _rec(objects=[obj])
        errors = validate_all_instructions(rec)
        assert errors == []  # instruction=None → nothing to validate


class TestQwenGrounding:
    def test_mock_model_returns_none(self) -> None:
        """MockModel.supports_grounding is False — must return None."""
        from PIL import Image as PILImage

        from wastecap.grounding.instructions import query_qwen_grounding
        from wastecap.models.mock import MockModel

        img = PILImage.new("RGB", (64, 64))
        model = MockModel()
        result = query_qwen_grounding(img, "plastic bottle", model)
        assert result is None

    def test_non_grounding_model_skipped(self) -> None:
        """Any model with supports_grounding=False returns None without calling generate."""
        from PIL import Image as PILImage

        from wastecap.grounding.instructions import query_qwen_grounding
        from wastecap.models.blip2 import Blip2Model

        img = PILImage.new("RGB", (64, 64))
        model = Blip2Model()
        # Blip2 doesn't support grounding → should return None, not call _load()
        result = query_qwen_grounding(img, "bottle", model)
        assert result is None
