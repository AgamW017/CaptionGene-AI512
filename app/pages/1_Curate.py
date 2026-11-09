"""
Curate page — browse images, select model/prompt, generate captions.

Layout
------
Sidebar: filters (source, split, status), pagination controls
Main:    left col = image + bboxes, right col = model selector + generate button
         Below: candidate cards (one per generated candidate)

State stored in st.session_state:
  curate_page        int  current page index (0-based)
  curate_source      str  filter
  curate_split       str  filter
  curate_status      str  filter
  curate_record_id   str  currently displayed image_id
  curate_editing     str  candidate text being edited (or "")
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_SRC_ROOT = str(Path(_PROJECT_ROOT) / "src")
for _path in (_PROJECT_ROOT, _SRC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.components.ui import inject_styles, page_header, section_heading  # noqa: E402

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Curate — WasteCap", page_icon="🖼️", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_db_path():  # type: ignore[return]
    from wastecap.store import init_db

    return init_db()


@st.cache_resource
def _get_model(backend: str):  # type: ignore[return]
    """Load (and cache) the caption model for this backend."""
    from wastecap.models.registry import get_model

    return get_model(backend)


def _ss(key: str, default=None):  # type: ignore[return]
    """Get a session_state key, initialising to default if absent."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    inject_styles()
    page_header(
        "Candidate studio",
        "Curate captions",
        "Browse the image queue, generate candidates, and keep every model output reviewable.",
    )

    db_path = _get_db_path()

    from wastecap.store import list_records

    # ── Sidebar filters ──────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Queue")
        source_opts = ["all", "taco", "zerowaste", "coco", "web"]
        split_opts = ["all", "train", "val", "test"]
        status_opts = ["all", "pending", "edited", "rejected", "approved"]

        sel_source = st.selectbox("Source", source_opts, key="curate_source")
        sel_split = st.selectbox("Split", split_opts, key="curate_split")
        sel_status = st.selectbox("Status", status_opts, key="curate_status")

        page_size = st.number_input("Images per page", min_value=1, max_value=50, value=5, step=1)
        st.divider()

        # Model selector
        st.subheader("Generation studio")
        backend = os.environ.get("WASTECAP_BACKEND", "mock")
        available_backends = ["mock", "blip2", "qwen_vl", "llava"]
        default_idx = available_backends.index(backend) if backend in available_backends else 0
        sel_model = st.selectbox("Model", available_backends, index=default_idx, key="curate_model")
        sel_level = st.selectbox(
            "Level",
            ["scene", "object"],
            key="curate_level",
            help="'scene' captions the whole image; 'object' crops each bbox.",
        )
        sel_n = st.number_input(
            "Candidates per image (n)", min_value=1, max_value=5, value=3, step=1
        )
        prompt_override = st.text_area(
            "Prompt template override",
            value="",
            help="Optional scene-level prompt. Leave blank to use the configured YAML template.",
        )
        uploaded = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])
        if uploaded is not None and st.button("Add uploaded image", key="add_upload"):
            from PIL import Image

            from wastecap.schema import AnnotationRecord, DataSource, RecordStatus, Split
            from wastecap.store import ingest_record

            upload_root = Path(_PROJECT_ROOT) / "data" / "processed" / "uploads"
            upload_root.mkdir(parents=True, exist_ok=True)
            suffix = Path(uploaded.name).suffix.lower() or ".png"
            image_id = f"upload_{hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]}"
            target = upload_root / f"{image_id}{suffix}"
            target.write_bytes(uploaded.getvalue())
            with Image.open(target) as image:
                width, height = image.size
            ingest_record(
                AnnotationRecord(
                    image_id=image_id,
                    file=str(target),
                    source=DataSource.web,
                    width=width,
                    height=height,
                    split=Split.train,
                    status=RecordStatus.pending,
                ),
                db_path,
            )
            st.success(f"Added `{image_id}` to the review queue.")
            st.rerun()

    # ── Load records ────────────────────────────────────────────────────────
    list_kwargs: dict = {"db_path": db_path}
    if sel_source != "all":
        list_kwargs["source"] = sel_source
    if sel_split != "all":
        list_kwargs["split"] = sel_split
    if sel_status != "all":
        list_kwargs["status"] = sel_status

    records = list_records(**list_kwargs)
    section_heading("Image queue", f"{len(records)} matching record(s)")

    if not records:
        st.info("No records match the current filters. Run `scripts/ingest.py` to load data.")
        return

    # ── Pagination ──────────────────────────────────────────────────────────
    total = len(records)
    n_pages = max(1, (total + page_size - 1) // page_size)
    page = _ss("curate_page", 0)
    page = max(0, min(page, n_pages - 1))

    col_prev, col_info, col_next = st.columns([1, 3, 1], vertical_alignment="center")
    with col_prev:
        if st.button("← Prev", disabled=page == 0, key="curate_prev"):
            st.session_state["curate_page"] = page - 1
            st.rerun()
    with col_info:
        st.markdown(f"**Page {page + 1} of {n_pages}**  ·  {total} records")
    with col_next:
        if st.button("Next →", disabled=page >= n_pages - 1, key="curate_next"):
            st.session_state["curate_page"] = page + 1
            st.rerun()

    page_records = records[page * page_size : (page + 1) * page_size]

    # ── Per-record display ──────────────────────────────────────────────────
    for rec in page_records:
        with st.expander(
            f"`{rec.image_id}` · {rec.source.value} · {rec.split.value if rec.split else '—'} "
            f"· status: **{rec.status.value}**",
            expanded=True,
        ):
            left, right = st.columns([1, 2])

            # ── Left: image + bboxes ──
            with left:
                from app.components.image_viewer import image_viewer, load_image

                img = load_image(rec.file)
                if img is not None:
                    bboxes = [o.bbox for o in rec.objects if o.bbox is not None]
                    labels = [
                        f"{o.object_id}: {o.category.value}"
                        for o in rec.objects
                        if o.bbox is not None
                    ]
                    image_viewer(
                        img,
                        bboxes=bboxes or None,
                        labels=labels or None,
                        caption=rec.image_id,
                    )
                    if bboxes:
                        st.dataframe(
                            [
                                {
                                    "Object": obj.object_id,
                                    "Class": obj.category.value,
                                    "Material": obj.material.value,
                                    "Box [x, y, w, h]": [round(v, 1) for v in obj.bbox],
                                }
                                for obj in rec.objects
                                if obj.bbox is not None
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                else:
                    st.warning(f"Image not on disk: `{rec.file}`")

                st.caption(
                    f"**{rec.width}×{rec.height}** · "
                    f"{len(rec.objects)} object(s) · "
                    f"{len(rec.candidates)} candidate(s)"
                )
                if rec.scene_context:
                    st.caption(f"🌍 {rec.scene_context}")

            # ── Right: generate + candidates ──
            with right:
                gen_key = f"gen_{rec.image_id}"
                st.markdown(f"**{rec.image_id}**")
                st.caption(
                    f"{rec.source.value} · {rec.split.value if rec.split else '—'} · "
                    f"{rec.status.value}"
                )
                if st.button(
                    ":material/auto_awesome: Generate candidates",
                    key=gen_key,
                    width="stretch",
                    type="primary",
                ):
                    if img is None:
                        st.error("Image file not found — cannot generate.")
                    else:
                        with st.spinner("Generating…"):
                            _run_generate(
                                rec,
                                img,
                                sel_model,
                                sel_level,
                                int(sel_n),
                                db_path,
                                prompt_override=prompt_override or None,
                            )
                        st.rerun()

                if not rec.candidates:
                    st.info("No candidates yet. Click Generate.")
                else:
                    from app.components.caption_card import caption_card

                    for ci, cand in enumerate(rec.candidates):
                        key_pfx = f"{rec.image_id}_c{ci}"
                        caption_card(cand, key_prefix=key_pfx, show_buttons=False)
                        # On Curate page, buttons are read-only — approve/reject
                        # happens only in Review.


def _run_generate(
    rec,
    img,
    model_name: str,
    level: str,
    n: int,
    db_path,
    *,
    prompt_override: str | None = None,
) -> None:
    """Generate candidates for *rec* and append to its candidates list in the DB."""
    from wastecap.cli import _generate_objects, _generate_scene
    from wastecap.models.registry import get_model
    from wastecap.store import update_record

    try:
        model = get_model(model_name)
        if level == "scene":
            new_cands = _generate_scene(model, img, model_name, n, prompt_override)
        else:
            new_cands = _generate_objects(model, img, rec, model_name, n)

        if new_cands:
            rec.candidates.extend(new_cands)
            update_record(rec, db_path)
            st.success(f"Generated {len(new_cands)} candidate(s).")
        else:
            st.warning("No candidates produced (no objects with bboxes for object-level?).")
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
        logger.exception("Generation error on %s", rec.image_id)


main()
