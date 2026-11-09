"""
Instructions page — view, edit and approve grounded sorting instructions.

For each image the page shows:
  - The image with ALL object bboxes drawn, then re-renders highlighting
    only the currently selected object
  - Per-object: auto-generated instruction text, editable textarea, approve button
  - A "Generate All" button that fills in instructions for all objects
    with valid bboxes
  - A "Qwen Grounding" button (visible when qwen_vl is the active backend)
    that queries the model for predicted boxes and stores them on the candidate

Approval rule (same as Review page):
  instruction is written to ObjectAnnotation.instruction and the record
  is saved via store.update_record. The instruction is only set here or
  via generate_all_instructions() + explicit save; never auto-set silently.

Validation:
  Every instruction must point to an object with a non-null bbox and a
  sorting_action before it can be saved.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_SRC_ROOT = str(Path(_PROJECT_ROOT) / "src")
for _path in (_PROJECT_ROOT, _SRC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.components.ui import inject_styles, page_header  # noqa: E402

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Instructions — WasteCap", page_icon="📋", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_db_path():  # type: ignore[return]
    from wastecap.store import init_db

    return init_db()


def _ss(key: str, default=None):  # type: ignore[return]
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _save_record(rec, db_path) -> None:
    from wastecap.store import update_record

    rec.updated_at = datetime.now(tz=UTC)
    update_record(rec, db_path)


# ---------------------------------------------------------------------------
# Image rendering helpers
# ---------------------------------------------------------------------------

# Colours matching image_viewer.py _BOX_COLOURS
_COLOURS = [
    "#FF4B4B",
    "#4B9CFF",
    "#4BFF7A",
    "#FFD24B",
    "#FF4BF0",
    "#4BFFF0",
    "#FF974B",
    "#C14BFF",
]
_HIGHLIGHT_COLOUR = "#FFFFFF"  # white outline for the selected object
_DIM_ALPHA = 80  # 0-255; other boxes are semi-transparent


def _render_image_with_highlight(
    img,
    objects,
    selected_idx: int | None,
) -> None:
    """Render image with all bboxes, highlighting the selected one."""
    from app.components.image_viewer import image_viewer
    from PIL import ImageDraw

    if not objects:
        image_viewer(img)
        return

    bboxes = [o.bbox for o in objects if o.bbox is not None]
    if not bboxes:
        image_viewer(img)
        return

    if selected_idx is None:
        labels = [f"{o.object_id}: {o.category.value}" for o in objects if o.bbox is not None]
        image_viewer(img, bboxes=bboxes, labels=labels)
        return

    # Draw all boxes dim, then redraw selected bold
    display = img.copy().convert("RGBA")
    overlay = display.copy()
    draw = ImageDraw.Draw(overlay)

    valid_objs = [(idx, o) for idx, o in enumerate(objects) if o.bbox is not None]
    for i, (object_idx, obj) in enumerate(valid_objs):
        x, y, w, h = obj.bbox
        colour = _COLOURS[i % len(_COLOURS)]
        width = 8 if object_idx == selected_idx else 3
        draw.rectangle([int(x), int(y), int(x + w), int(y + h)], outline=colour, width=width)

    # Convert back to RGB for st.image
    import PIL.Image

    result = PIL.Image.alpha_composite(display, overlay).convert("RGB")
    st.image(result, use_container_width=True)
    st.caption(
        "Selected object is outlined in white; colored outlines are the other reference boxes."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    inject_styles()
    page_header(
        "Grounded actions",
        "Sorting instructions",
        "Select an object, verify its highlighted box, and approve a precise sorting action.",
    )

    db_path = _get_db_path()

    from wastecap.store import list_records

    # ── Sidebar filters ──────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        source_opts = ["all", "taco", "zerowaste", "coco"]
        split_opts = ["all", "train", "val", "test"]
        sel_source = st.selectbox("Source", source_opts, key="instr_source")
        sel_split = st.selectbox("Split", split_opts, key="instr_split")
        only_with_objects = st.checkbox(
            "Only records with objects",
            value=True,
            key="instr_only_objs",
        )
        only_needs_instruction = st.checkbox(
            "Only objects needing instructions",
            value=False,
            key="instr_only_needs",
        )
        page_size = st.number_input("Page size", 1, 20, 5, step=1)

    # ── Load records ─────────────────────────────────────────────────────────
    list_kwargs: dict = {"db_path": db_path}
    if sel_source != "all":
        list_kwargs["source"] = sel_source
    if sel_split != "all":
        list_kwargs["split"] = sel_split

    records = list_records(**list_kwargs)

    if only_with_objects:
        records = [r for r in records if r.objects]

    if not records:
        st.info("No records match the current filters. Run ingest first.")
        return

    # ── Pagination ────────────────────────────────────────────────────────────
    n_pages = max(1, (len(records) + page_size - 1) // page_size)
    page = _ss("instr_page", 0)
    page = max(0, min(page, n_pages - 1))

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Prev", disabled=page == 0, key="instr_prev"):
            st.session_state["instr_page"] = page - 1
            st.rerun()
    with col_info:
        st.caption(f"Page {page + 1} / {n_pages}  ·  {len(records)} records")
    with col_next:
        if st.button("Next →", disabled=page >= n_pages - 1, key="instr_next"):
            st.session_state["instr_page"] = page + 1
            st.rerun()

    page_records = records[page * page_size : (page + 1) * page_size]

    # ── Per-record display ────────────────────────────────────────────────────
    for rec in page_records:
        objects_needing = [o for o in rec.objects if o.instruction is None and o.bbox is not None]

        if only_needs_instruction and not objects_needing:
            continue

        n_done = sum(1 for o in rec.objects if o.instruction is not None)
        n_total = len(rec.objects)

        st.markdown(
            f"---\n**`{rec.image_id}`** · {rec.source.value} · instructions: **{n_done}/{n_total}**"
        )
        with st.container(border=True):
            left, right = st.columns([1, 2])

            with left:
                from app.components.image_viewer import load_image

                img = load_image(rec.file)
                selected_obj_idx = _ss(f"instr_sel_{rec.image_id}", None)

                if img is not None:
                    _render_image_with_highlight(img, rec.objects, selected_obj_idx)
                    st.caption(f"{rec.width}×{rec.height} · {len(rec.objects)} object(s)")
                else:
                    st.warning(f"Image not found: `{rec.file}`")

                # ── Generate All button ──
                if st.button(
                    "⚡ Generate all instructions",
                    key=f"instr_genall_{rec.image_id}",
                    use_container_width=True,
                    help=(
                        "Auto-generate instructions for all objects with a bbox and sorting_action."
                    ),
                ):
                    from wastecap.grounding.instructions import generate_all_instructions

                    updated = generate_all_instructions(rec, rec.width, rec.height)
                    n_generated = sum(
                        1
                        for a, b in zip(rec.objects, updated.objects, strict=False)
                        if a.instruction is None and b.instruction is not None
                    )
                    if n_generated > 0:
                        _save_record(updated, db_path)
                        st.success(f"Generated {n_generated} instruction(s).")
                        st.rerun()
                    else:
                        st.info(
                            "No new instructions generated. "
                            "Objects may be missing bbox or sorting_action."
                        )

            with right:
                if not rec.objects:
                    st.info("No objects annotated for this image.")
                    continue

                st.subheader("Per-object instructions")

                for oi, obj in enumerate(rec.objects):
                    status_icon = "✅" if obj.instruction else "○"
                    cond_str = f"/{obj.condition.value}" if obj.condition else ""
                    st.markdown(
                        f"**`{obj.object_id}`** — "
                        f"{obj.category.value}/{obj.material.value}{cond_str} {status_icon}"
                    )
                    with st.container(border=True):
                        # Click to highlight
                        if st.button(
                            "🎯 Highlight on image",
                            key=f"instr_hl_{rec.image_id}_{oi}",
                        ):
                            st.session_state[f"instr_sel_{rec.image_id}"] = oi
                            st.rerun()

                        # Validation status
                        from wastecap.grounding.instructions import validate_instruction

                        valid, reason = validate_instruction(obj, rec.width, rec.height)

                        if not valid:
                            st.warning(f"⚠ Cannot approve: {reason}")
                        else:
                            if obj.bbox:
                                from wastecap.grounding.boxes import spatial_description

                                loc = spatial_description(obj.bbox, rec.width, rec.height)
                                action_str = obj.sorting_action.value if obj.sorting_action else "—"
                                st.caption(
                                    f"bbox: {[round(v) for v in obj.bbox]}  ·  "
                                    f"location: **{loc}**  ·  "
                                    f"action: **{action_str}**"
                                )

                        # Auto-generate suggestion
                        if valid and obj.instruction is None:
                            from wastecap.grounding.instructions import generate_instruction

                            try:
                                suggestion = generate_instruction(obj, rec.width, rec.height)
                            except ValueError:
                                suggestion = ""
                        else:
                            suggestion = obj.instruction or ""

                        # Edit textarea
                        new_text = st.text_area(
                            "Instruction text",
                            value=suggestion,
                            height=100,
                            key=f"instr_text_{rec.image_id}_{oi}",
                            disabled=not valid,
                            placeholder="Cannot generate — set bbox and sorting_action first.",
                        )

                        # Save + approve
                        col_save, col_clear = st.columns(2)
                        with col_save:
                            if st.button(
                                "✅ Approve instruction",
                                key=f"instr_approve_{rec.image_id}_{oi}",
                                disabled=not valid or not new_text.strip(),
                                use_container_width=True,
                                type="primary",
                            ):
                                # Final validation before write
                                if obj.bbox is None:
                                    st.error("Cannot approve: bbox is required.")
                                else:
                                    updated_obj = obj.model_copy(
                                        update={"instruction": new_text.strip()}
                                    )
                                    valid_updated, validation_error = validate_instruction(
                                        updated_obj, rec.width, rec.height
                                    )
                                    if not valid_updated:
                                        st.error(f"Cannot approve: {validation_error}")
                                        continue
                                    # Schema validator enforces bbox present
                                    new_objects = list(rec.objects)
                                    new_objects[oi] = updated_obj
                                    updated_rec = rec.model_copy(update={"objects": new_objects})
                                    _save_record(updated_rec, db_path)
                                    st.success("Instruction approved and saved.")
                                    st.rerun()

                        with col_clear:
                            if obj.instruction and st.button(
                                "✖ Clear",
                                key=f"instr_clear_{rec.image_id}_{oi}",
                                use_container_width=True,
                            ):
                                updated_obj = obj.model_copy(update={"instruction": None})
                                new_objects = list(rec.objects)
                                new_objects[oi] = updated_obj
                                updated_rec = rec.model_copy(update={"objects": new_objects})
                                _save_record(updated_rec, db_path)
                                st.info("Instruction cleared.")
                                st.rerun()

                        # ── Qwen grounding (only if qwen_vl backend) ──
                        backend = os.environ.get("WASTECAP_BACKEND", "mock")
                        if backend == "qwen_vl" and img is not None and obj.bbox is not None:
                            from wastecap.grounding.boxes import spatial_description

                            cond_str = obj.condition.value + " " if obj.condition else ""
                            mat_str = (
                                obj.material.value + " " if obj.material.value != "unknown" else ""
                            )
                            obj_desc = f"{cond_str}{mat_str}{obj.category.value}"
                            if st.button(
                                f"🔍 Query Qwen grounding: '{obj_desc}'",
                                key=f"instr_qwen_{rec.image_id}_{oi}",
                            ):
                                from wastecap.grounding.instructions import query_qwen_grounding
                                from wastecap.models.registry import get_model

                                with st.spinner("Querying Qwen2.5-VL…"):
                                    model = get_model("qwen_vl")
                                    boxes = query_qwen_grounding(
                                        img, obj_desc, model, rec.width, rec.height
                                    )
                                if boxes:
                                    st.success(
                                        f"Qwen returned {len(boxes)} predicted box(es): "
                                        f"{[[round(v) for v in b] for b in boxes[:3]]}"
                                    )
                                    st.caption(
                                        "These are stored for IoU evaluation in M7. "
                                        "They are NOT ground truth."
                                    )
                                else:
                                    st.info("Qwen returned no grounding boxes.")


main()
