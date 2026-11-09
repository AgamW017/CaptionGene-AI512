"""
Review page — approve / reject / edit annotation queue.

M4 additions
------------
- Sidebar filter: model (any candidate model), min/max ReviewScore, flagged only
- Score badge shown on each candidate card (composite, grade, parse, length)
- Hallucination hints shown per candidate (expandable)
- "Flag as hallucination" toggle on the Review entry

This is the ONLY place in the app that sets status = 'approved'.
CLAUDE.md: "status = approved is set only by a human action in the UI."
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

st.set_page_config(page_title="Review — WasteCap", page_icon="✅", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_db_path():  # type: ignore[return]
    from wastecap.store import init_db

    return init_db()


def _reviewer() -> str:
    return os.environ.get("WASTECAP_REVIEWER", "annotator_1")


def _ss(key: str, default=None):  # type: ignore[return]
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---------------------------------------------------------------------------
# DB actions — ALL writes go through store.update_record
# ---------------------------------------------------------------------------


def _approve(rec, candidate_text: str, hallucination: bool, db_path) -> None:
    from wastecap.schema import RecordStatus, Review
    from wastecap.store import update_record

    rev = Review(
        reviewer=_reviewer(),
        notes="",
        hallucination=hallucination,
        reviewed_at=datetime.now(tz=UTC),
    )
    rec.status = RecordStatus.approved
    rec.caption = candidate_text.strip()
    rec.review = rev
    rec.updated_at = datetime.now(tz=UTC)
    update_record(rec, db_path)


def _reject(rec, db_path) -> None:
    from wastecap.schema import RecordStatus
    from wastecap.store import update_record

    rec.status = RecordStatus.rejected
    rec.updated_at = datetime.now(tz=UTC)
    update_record(rec, db_path)


def _save_edit(rec, cand_idx: int, new_text: str, db_path) -> None:
    from wastecap.schema import RecordStatus
    from wastecap.store import update_record

    rec.candidates[cand_idx] = rec.candidates[cand_idx].model_copy(
        update={"text": new_text.strip()}
    )
    rec.status = RecordStatus.edited
    rec.updated_at = datetime.now(tz=UTC)
    update_record(rec, db_path)


def _save_structured(rec, obj_idx: int, updated_obj, scene_ctx: str | None, db_path) -> None:
    from wastecap.store import update_record

    if updated_obj is not None and obj_idx < len(rec.objects):
        rec.objects[obj_idx] = updated_obj
    if scene_ctx is not None:
        rec.scene_context = scene_ctx
    rec.updated_at = datetime.now(tz=UTC)
    update_record(rec, db_path)


# ---------------------------------------------------------------------------
# Score badge (shared with Compare page but kept local to avoid coupling)
# ---------------------------------------------------------------------------


def _score_line(score) -> str:
    grade_colour = {"good": "green", "marginal": "orange", "poor": "red"}
    colour = grade_colour.get(score.grade, "grey")
    clip_str = f"{score.clip_score:.3f}" if score.clip_score is not None else "—"
    return (
        f":{colour}[{score.grade.upper()} {score.composite:.2f}]  "
        f"CLIP={clip_str}  compl={score.completeness:.0%}  "
        f"parse={'✓' if score.parse_ok else '✗'}  "
        f"len={'✓' if score.length_ok else '✗'}"
    )


# ---------------------------------------------------------------------------
# Candidate rendering with score + hallucination hints
# ---------------------------------------------------------------------------


def _render_candidate(rec, ci: int, cand, edit_idx: int, db_path) -> str | None:
    """Render one candidate card. Returns action taken or None."""
    from wastecap.quality.hallucination import (
        flag_candidate,
        get_hallucination_flags,
    )
    from wastecap.quality.scoring import score_candidate

    score = score_candidate(cand)
    flagged_cand = flag_candidate(cand, rec.objects)
    h_flags = get_hallucination_flags(flagged_cand)

    key_pfx = f"rv_{rec.image_id}_c{ci}"
    is_editing = edit_idx == ci

    with st.container(border=True):
        # Header row
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            st.caption(f"🤖 `{cand.model}` · `{cand.prompt_id}`")
        with hcol2:
            st.markdown(_score_line(score))

        # Hallucination hints
        if h_flags:
            with st.expander(f"⚠️ {len(h_flags)} hint(s)", expanded=False):
                for flag in h_flags:
                    st.warning(f"• {flag}", icon="⚠️")
                st.caption("Heuristic hints — reviewer makes the final call.")

        # Parse-failure warning
        if cand.parse_failed:
            st.error("⚠️ Structured parse failed.")

        # Caption text
        st.markdown(f"> {cand.text[:500]}")

        # Buttons
        b1, b2, b3 = st.columns(3)
        action: str | None = None
        with b1:
            if st.button(
                "✅ Approve", key=f"{key_pfx}_approve", use_container_width=True, type="primary"
            ):
                action = "approve"
        with b2:
            if st.button("❌ Reject", key=f"{key_pfx}_reject", use_container_width=True):
                action = "reject"
        with b3:
            if st.button("✏️ Edit", key=f"{key_pfx}_edit", use_container_width=True):
                action = "edit"

        # Inline edit
        if is_editing:
            new_text = st.text_area(
                "Edit caption text",
                value=cand.text,
                height=120,
                key=f"{key_pfx}_edit_area",
            )
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("💾 Save edit", key=f"{key_pfx}_save_edit", use_container_width=True):
                    _save_edit(rec, ci, new_text, db_path)
                    st.session_state["review_edit_idx"] = -1
                    st.success("Edit saved.")
                    st.rerun()
            with sc2:
                if st.button("✖ Cancel", key=f"{key_pfx}_cancel", use_container_width=True):
                    st.session_state["review_edit_idx"] = -1
                    st.rerun()

    return action


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    inject_styles()
    page_header(
        "Human quality gate",
        "Review queue",
        "Edit, flag, approve, or reject candidates with a clear audit trail.",
    )

    db_path = _get_db_path()

    from wastecap.quality.filters import FilterSpec, apply_filters
    from wastecap.store import list_records

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Queue filters")

        status_opts = ["pending", "edited", "all", "approved", "rejected"]
        sel_status = st.selectbox("Status", status_opts, index=0, key="review_status")
        source_opts = ["all", "taco", "zerowaste", "coco", "web"]
        sel_source = st.selectbox("Source", source_opts, key="review_source")
        split_opts = ["all", "train", "val", "test"]
        sel_split = st.selectbox("Split", split_opts, key="review_split")

        st.divider()
        st.subheader("Quality filters")

        model_opts = ["all", "mock", "blip2", "qwen_vl", "llava"]
        sel_model = st.selectbox(
            "Model",
            model_opts,
            key="review_model",
            help="Filter to records with a candidate from this model.",
        )

        min_score = st.slider("Min composite score", 0.0, 1.0, 0.0, 0.05, key="review_min_score")
        max_score = st.slider("Max composite score", 0.0, 1.0, 1.0, 0.05, key="review_max_score")
        only_flagged = st.checkbox("Only hallucination-flagged", key="review_only_flagged")
        only_parse_failed = st.checkbox("Only parse-failed candidates", key="review_parse_fail")

        st.divider()
        st.caption("Approval is permanent. Reject to skip.")

    # ── Load all, then apply in-memory filters ────────────────────────────
    list_kwargs: dict = {"db_path": db_path}
    if sel_status != "all":
        list_kwargs["status"] = sel_status
    if sel_source != "all":
        list_kwargs["source"] = sel_source
    if sel_split != "all":
        list_kwargs["split"] = sel_split

    records = list_records(**list_kwargs)

    # Apply quality filters
    spec = FilterSpec(
        models=[sel_model] if sel_model != "all" else [],
        min_score=min_score if min_score > 0.0 else None,
        max_score=max_score if max_score < 1.0 else None,
        only_flagged=only_flagged,
        only_parse_failed=only_parse_failed,
    )
    queue = apply_filters(records, spec)

    if not queue:
        st.success("Queue is empty — nothing to review with current filters. 🎉")
        return

    # ── Navigation ────────────────────────────────────────────────────────
    idx = _ss("review_idx", 0)
    idx = max(0, min(idx, len(queue) - 1))

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Prev", disabled=idx == 0, key="review_prev"):
            st.session_state["review_idx"] = idx - 1
            st.session_state["review_edit_idx"] = -1
            st.rerun()
    with col_info:
        st.caption(f"Record {idx + 1} / {len(queue)}")
    with col_next:
        if st.button("Next →", disabled=idx >= len(queue) - 1, key="review_next"):
            st.session_state["review_idx"] = idx + 1
            st.session_state["review_edit_idx"] = -1
            st.rerun()

    rec = queue[idx]
    edit_idx = _ss("review_edit_idx", -1)
    hal_checked = _ss(f"hal_check_{rec.image_id}", False)

    # ── Main layout ────────────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        from app.components.image_viewer import image_viewer, load_image

        img = load_image(rec.file)
        if img is not None:
            bboxes = [o.bbox for o in rec.objects if o.bbox is not None]
            labels = [
                f"{o.object_id}: {o.category.value}" for o in rec.objects if o.bbox is not None
            ]
            image_viewer(img, bboxes=bboxes or None, labels=labels or None, caption=rec.image_id)
        else:
            st.warning(f"Image not found: `{rec.file}`")

        st.caption(
            f"**{rec.image_id}** · {rec.source.value} · "
            f"split: {rec.split.value if rec.split else '—'}"
        )
        st.caption(f"Status: **{rec.status.value}**")
        if rec.caption:
            st.info(f"📝 Approved caption: {rec.caption}")

        # ── Object structured fields ──
        if rec.objects:
            st.subheader("Objects")
            for oi, obj in enumerate(rec.objects):
                label = f"`{obj.object_id}` {obj.category.value}/{obj.material.value}"
                if obj.condition:
                    label += f"/{obj.condition.value}"
                with st.expander(label):
                    from app.components.structured_form import structured_form

                    updated_obj, new_scene = structured_form(
                        obj,
                        key_prefix=f"sf_{rec.image_id}_o{oi}",
                        scene_context=rec.scene_context if oi == 0 else None,
                    )
                    if updated_obj is not None or new_scene is not None:
                        _save_structured(rec, oi, updated_obj, new_scene, db_path)
                        st.success("Saved.")
                        st.rerun()
        else:
            from app.components.structured_form import structured_form

            _, new_scene = structured_form(
                None,
                key_prefix=f"sf_{rec.image_id}_scene",
                scene_context=rec.scene_context,
            )
            if new_scene is not None:
                _save_structured(rec, 0, None, new_scene, db_path)
                st.success("Scene context saved.")
                st.rerun()

    with right:
        st.subheader("Candidates")

        # Hallucination verdict checkbox (human decision)
        new_hal = st.checkbox(
            "🚩 Mark approved caption as hallucinated",
            value=hal_checked,
            key=f"hal_{rec.image_id}",
            help=(
                "Check this before Approve if the caption names objects or "
                "materials not visible in the image. This sets "
                "review.hallucination=True on the Review entry."
            ),
        )
        st.session_state[f"hal_check_{rec.image_id}"] = new_hal

        if not rec.candidates:
            st.info(
                "No candidates yet. Go to Curate page, generate with a model, then return here."
            )
        else:
            for ci, cand in enumerate(rec.candidates):
                action = _render_candidate(rec, ci, cand, edit_idx, db_path)

                if action == "approve":
                    _approve(rec, cand.text, new_hal, db_path)
                    st.success(f"✅ Approved (hallucination={new_hal}).")
                    st.session_state["review_idx"] = min(idx + 1, len(queue) - 1)
                    st.session_state["review_edit_idx"] = -1
                    st.rerun()
                elif action == "reject":
                    _reject(rec, db_path)
                    st.warning("❌ Rejected.")
                    st.session_state["review_idx"] = min(idx + 1, len(queue) - 1)
                    st.session_state["review_edit_idx"] = -1
                    st.rerun()
                elif action == "edit":
                    st.session_state["review_edit_idx"] = ci
                    st.rerun()

        st.divider()
        if st.button(
            "❌ Reject record (no suitable candidate)",
            key=f"rv_{rec.image_id}_reject_all",
            use_container_width=True,
        ):
            _reject(rec, db_path)
            st.session_state["review_idx"] = min(idx + 1, len(queue) - 1)
            st.session_state["review_edit_idx"] = -1
            st.rerun()


main()
