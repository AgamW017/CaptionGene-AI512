"""
Review page — approve / reject / edit annotation queue.

This is the ONLY place in the app that sets status = 'approved'.
CLAUDE.md rule: "status = approved is set only by a human action in the UI."

Layout
------
Sidebar: queue filters (status filter defaults to 'pending'),
         batch controls
Main:    one record at a time
         left col = image + object list
         right col = candidate cards (approve/reject/edit) + structured form

State
-----
  review_idx       int   position in the filtered queue
  review_edit_idx  int   index of candidate being edited (-1 = none)
  review_edit_text str   current text in the edit textarea
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import streamlit as st

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
# Actions — all DB writes go through store.update_record
# ---------------------------------------------------------------------------


def _approve(rec, candidate_text: str, db_path) -> None:
    """Set record status = approved with the selected candidate as the caption."""
    from wastecap.schema import RecordStatus, Review
    from wastecap.store import update_record

    rev = Review(
        reviewer=_reviewer(),
        notes="",
        hallucination=False,
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
    """Overwrite candidate text and set record status = edited."""
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
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("✅ Review")

    db_path = _get_db_path()

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
        st.caption("Approval is permanent in the DB. Reject to skip.")

    # ── Load queue ───────────────────────────────────────────────────────────
    list_kwargs: dict = {"db_path": db_path}
    if sel_status != "all":
        list_kwargs["status"] = sel_status
    if sel_source != "all":
        list_kwargs["source"] = sel_source
    if sel_split != "all":
        list_kwargs["split"] = sel_split

    queue = list_records(**list_kwargs)

    if not queue:
        st.success("Queue is empty — nothing to review with current filters. 🎉")
        return

    # ── Navigation ───────────────────────────────────────────────────────────
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

    # ── Main layout ──────────────────────────────────────────────────────────
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

        # ── Object list ──
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
            # No objects — still show scene context form
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

        if not rec.candidates:
            st.info("No candidates for this record. Go to the Curate page and run Generate first.")
        else:
            from app.components.caption_card import caption_card

            for ci, cand in enumerate(rec.candidates):
                key_pfx = f"rv_{rec.image_id}_c{ci}"
                action = caption_card(
                    cand,
                    key_prefix=key_pfx,
                    show_buttons=True,
                    selected=(edit_idx == ci),
                )

                if action == "approve":
                    _approve(rec, cand.text, db_path)
                    st.success(f"✅ Approved! Caption set to candidate {ci}.")
                    st.session_state["review_idx"] = min(idx + 1, len(queue) - 1)
                    st.session_state["review_edit_idx"] = -1
                    st.rerun()

                elif action == "reject":
                    _reject(rec, db_path)
                    st.warning("❌ Record rejected.")
                    st.session_state["review_idx"] = min(idx + 1, len(queue) - 1)
                    st.session_state["review_edit_idx"] = -1
                    st.rerun()

                elif action == "edit":
                    st.session_state["review_edit_idx"] = ci
                    st.rerun()

                # ── Inline edit textarea ──
                if edit_idx == ci:
                    new_text = st.text_area(
                        "Edit caption text",
                        value=cand.text,
                        height=120,
                        key=f"{key_pfx}_edit_area",
                    )
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button(
                            "💾 Save edit", key=f"{key_pfx}_save_edit", use_container_width=True
                        ):
                            _save_edit(rec, ci, new_text, db_path)
                            st.session_state["review_edit_idx"] = -1
                            st.success("Edit saved.")
                            st.rerun()
                    with col_cancel:
                        if st.button(
                            "✖ Cancel", key=f"{key_pfx}_cancel_edit", use_container_width=True
                        ):
                            st.session_state["review_edit_idx"] = -1
                            st.rerun()

        # ── Reject whole record without candidate ──
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
