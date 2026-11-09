"""Stats page — dataset statistics dashboard."""

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_SRC_ROOT = str(Path(_PROJECT_ROOT) / "src")
for _path in (_PROJECT_ROOT, _SRC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.components.ui import inject_styles, page_header, section_heading  # noqa: E402

from wastecap.stats import compute_statistics  # noqa: E402

st.set_page_config(page_title="Stats — WasteCap", page_icon="📊")
inject_styles()
page_header(
    "Observability",
    "Dataset health",
    "See coverage, taxonomy balance, and human review progress at a glance.",
)

stats = compute_statistics()
metric_cols = st.columns(4)
metric_cols[0].metric("Images", stats["images"])
metric_cols[1].metric("Approved", stats["approved_images"])
metric_cols[2].metric("Approval rate", f"{stats['approval_rate']:.1%}")
metric_cols[3].metric("Reviewed", stats["review_progress"]["reviewed_images"])

section_heading("Coverage", "Distribution across the current source of truth")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Images by split**")
    st.dataframe(
        [{"split": name, "images": count} for name, count in stats["images_by_split"].items()],
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Review progress**")
    progress = stats["review_progress"]
    st.progress(progress["review_rate"], text=f"{progress['review_rate']:.1%} reviewed")
    st.json(progress["status_counts"])
with col2:
    st.markdown("**Class distribution**")
    st.dataframe(
        [{"class": name, "images": count} for name, count in stats["class_distribution"].items()],
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Captions per class**")
    st.dataframe(
        [{"class": name, "captions": count} for name, count in stats["captions_per_class"].items()],
        hide_index=True,
        width="stretch",
    )

section_heading("Model quality", "Reference-free candidate scoring")
clip_scores = stats["clipscore_distribution"]
if clip_scores:
    st.bar_chart({"CLIPScore": clip_scores}, horizontal=False)
    st.caption(f"{len(clip_scores)} scored candidate(s)")
else:
    st.info("No CLIPScore values are stored yet. Run the CLIPScore batch command after generation.")
