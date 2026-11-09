"""
Compare page — side-by-side BLIP-2 vs Qwen2.5-VL outputs.

For each image the page shows:
  - The image with bboxes
  - One column per model (best candidate by composite ReviewScore)
  - ReviewScore breakdown: CLIPScore, completeness, parse, length, grade
  - Hallucination flags (hints, not verdicts)
  - A "Score all candidates" button that computes CLIPScore lazily and
    writes it back to the DB

The page is read-only. To approve/reject, go to the Review page.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.components.ui import inject_styles, page_header  # noqa: E402

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Compare — WasteCap", page_icon="⚖️", layout="wide")

_COMPARE_MODELS = ["blip2", "qwen_vl"]


@st.cache_resource
def _get_db_path():  # type: ignore[return]
    from wastecap.store import init_db

    return init_db()


def _ss(key: str, default=None):  # type: ignore[return]
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---------------------------------------------------------------------------
# Score badge helper
# ---------------------------------------------------------------------------


def _score_badge(score) -> str:  # type: ignore[return]
    """Return a coloured markdown badge string for a ReviewScore."""
    grade_colour = {"good": "green", "marginal": "orange", "poor": "red"}
    colour = grade_colour.get(score.grade, "grey")
    clip_str = f"{score.clip_score:.3f}" if score.clip_score is not None else "—"
    return (
        f":{colour}[**{score.grade.upper()}** · {score.composite:.2f}]  "
        f"CLIP={clip_str}  compl={score.completeness:.0%}  "
        f"parse={'✓' if score.parse_ok else '✗'}  "
        f"len={'✓' if score.length_ok else '✗'}"
    )


# ---------------------------------------------------------------------------
# Per-model column
# ---------------------------------------------------------------------------


def _render_model_col(model_name: str, candidate, score) -> None:
    """Render one model's best candidate in its column."""
    from wastecap.quality.hallucination import get_hallucination_flags

    st.subheader(f"🤖 {model_name}")

    if candidate is None:
        st.info(
            f"No candidate from **{model_name}** yet.  \n"
            "Go to the Curate page, select this model, and click Generate."
        )
        return

    # Score badge
    st.markdown(_score_badge(score))

    # Hallucination flags
    flags = get_hallucination_flags(candidate)
    if flags:
        with st.container(border=True):
            st.markdown(f"⚠️ **{len(flags)} hallucination hint(s)**")
            for f in flags:
                st.warning(f"• {f}", icon="⚠️")
            st.caption("These are heuristic hints — the reviewer makes the final call.")

    # Caption text
    st.markdown("**Caption**")
    if candidate.parse_failed:
        st.error("⚠️ Parse failed — raw model output shown below.")
    st.markdown(f"> {candidate.text[:600]}")

    # Structured fields if available
    if candidate.raw_parsed and not candidate.parse_failed:
        objects = candidate.raw_parsed.get("objects") or []
        if objects and isinstance(objects, list):
            with st.container(border=True):
                st.markdown("**Structured fields**")
                obj = objects[0]
                cols = st.columns(3)
                cols[0].metric("Category", obj.get("category", "—"))
                cols[1].metric("Material", obj.get("material", "—"))
                cols[2].metric("Condition", obj.get("condition", "—"))
                st.caption(f"count={obj.get('count', '—')}  location={obj.get('location', '—')}")
                scene = candidate.raw_parsed.get("scene_context", "")
                if scene:
                    st.caption(f"🌍 {scene}")

    st.caption(f"`{candidate.prompt_id}` · {candidate.created_at.strftime('%Y-%m-%d %H:%M')}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    inject_styles()
    page_header(
        "Model lab",
        "Compare candidates",
        "Inspect BLIP-2 and Qwen2.5-VL side by side. Approval happens in Review.",
    )

    db_path = _get_db_path()

    from wastecap.store import list_records

    # ── Sidebar filters ──────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        source_opts = ["all", "taco", "zerowaste", "coco"]
        split_opts = ["all", "train", "val", "test"]
        status_opts = ["all", "pending", "edited", "approved", "rejected"]

        sel_source = st.selectbox("Source", source_opts, key="cmp_source")
        sel_split = st.selectbox("Split", split_opts, key="cmp_split")
        sel_status = st.selectbox("Status", status_opts, key="cmp_status")
        only_both = st.checkbox(
            "Only records with both models",
            value=False,
            key="cmp_both",
            help="Hide records where one model hasn't generated yet.",
        )
        page_size = st.number_input("Page size", min_value=1, max_value=20, value=5, step=1)

    # ── Load ─────────────────────────────────────────────────────────────────
    list_kwargs: dict = {"db_path": db_path}
    if sel_source != "all":
        list_kwargs["source"] = sel_source
    if sel_split != "all":
        list_kwargs["split"] = sel_split
    if sel_status != "all":
        list_kwargs["status"] = sel_status

    records = list_records(**list_kwargs)

    if not records:
        st.info("No records match the current filters. Run ingest + generate first.")
        return

    # Filter to records with at least one of the compare models
    records = [r for r in records if any(c.model in _COMPARE_MODELS for c in r.candidates)]

    if only_both:
        records = [
            r
            for r in records
            if all(any(c.model == m for c in r.candidates) for m in _COMPARE_MODELS)
        ]

    if not records:
        st.info(
            "No records have candidates from the compare models yet. "
            "Go to the Curate page and generate with blip2 / qwen_vl."
        )
        return

    # ── Pagination ────────────────────────────────────────────────────────
    n_pages = max(1, (len(records) + page_size - 1) // page_size)
    page = _ss("cmp_page", 0)
    page = max(0, min(page, n_pages - 1))

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Prev", disabled=page == 0, key="cmp_prev"):
            st.session_state["cmp_page"] = page - 1
            st.rerun()
    with col_info:
        st.caption(f"Page {page + 1} / {n_pages}  ·  {len(records)} records with model output")
    with col_next:
        if st.button("Next →", disabled=page >= n_pages - 1, key="cmp_next"):
            st.session_state["cmp_page"] = page + 1
            st.rerun()

    page_records = records[page * page_size : (page + 1) * page_size]

    # ── Per-record compare view ───────────────────────────────────────────
    from wastecap.quality.clipscore import score_candidates_for_record
    from wastecap.quality.filters import best_candidate_per_model
    from wastecap.quality.hallucination import flag_all_candidates
    from wastecap.quality.scoring import score_candidate

    for rec in page_records:
        with st.expander(
            f"`{rec.image_id}` · {rec.source.value} · **{rec.status.value}**",
            expanded=True,
        ):
            # ── Score CLIPScore lazily ──
            unscored = sum(1 for c in rec.candidates if c.clip_score is None)
            if unscored > 0 and st.button(
                f"🔢 Compute CLIPScore ({unscored} unscored)",
                key=f"score_{rec.image_id}",
            ):
                n = score_candidates_for_record(rec, db_path)
                if n > 0:
                    st.success(f"Scored {n} candidate(s).")
                else:
                    st.info(
                        "CLIP model not available. "
                        "Install `.[ml]` extras and set WASTECAP_DEVICE=cuda."
                    )
                st.rerun()

            # Apply hallucination flags (in-memory only)
            rec_flagged = flag_all_candidates(rec)
            best_per_model = best_candidate_per_model(rec_flagged.candidates)

            img_col, blip_col, qwen_col = st.columns([1, 1, 1])

            with img_col:
                from app.components.image_viewer import image_viewer, load_image

                img = load_image(rec.file)
                if img is not None:
                    bboxes = [o.bbox for o in rec.objects if o.bbox is not None]
                    labels = [
                        f"{o.object_id}: {o.category.value}"
                        for o in rec.objects
                        if o.bbox is not None
                    ]
                    image_viewer(img, bboxes=bboxes or None, labels=labels or None)
                    st.caption(
                        f"Reference annotations: {len(bboxes)} box(es). "
                        "Model boxes are reported separately below."
                    )
                else:
                    st.warning("Image not found on disk.")

                st.caption(
                    f"{rec.width}×{rec.height} · "
                    f"{len(rec.objects)} obj(s) · "
                    f"{len(rec.candidates)} candidates"
                )

            with blip_col:
                cand = best_per_model.get("blip2")
                score = score_candidate(cand) if cand else None
                _render_model_col("blip2", cand, score)

            with qwen_col:
                cand = best_per_model.get("qwen_vl")
                score = score_candidate(cand) if cand else None
                _render_model_col("qwen_vl", cand, score)


main()
