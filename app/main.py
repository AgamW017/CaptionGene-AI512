"""WasteCap Streamlit app entry point.

Run with:
    streamlit run app/main.py

Environment variables (see .env.example):
    WASTECAP_BACKEND   mock | blip2 | qwen_vl | llava
    WASTECAP_DEVICE    cpu | cuda
    WASTECAP_QUANTIZE  none | 4bit | 8bit
    WASTECAP_DB_PATH   path to SQLite DB
    WASTECAP_REVIEWER  reviewer name
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Support `streamlit run app/main.py` directly from a source checkout without
# requiring a prior editable install.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
for _path in (str(_PROJECT_ROOT), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Must be first Streamlit call.
st.set_page_config(
    page_title="WasteCap",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up logging for the app (level from env var).
logging.basicConfig(
    level=os.environ.get("WASTECAP_LOG_LEVEL", "INFO"),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@st.cache_resource
def _init_db() -> Path:
    """Initialise the database once per app session."""
    from wastecap.store import init_db

    return init_db()


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="wc-brand"><div class="wc-brand-mark">W</div>'
            '<div><div class="wc-brand-name">WasteCap</div>'
            '<div class="wc-brand-sub">Evidence-first curation</div></div></div>',
            unsafe_allow_html=True,
        )
        st.caption("A focused workspace for turning waste images into verified, grounded data.")
        st.divider()

        # Environment / config info
        st.subheader("Runtime")
        backend = os.environ.get("WASTECAP_BACKEND", "mock")
        device = os.environ.get("WASTECAP_DEVICE", "cpu")
        quantize = os.environ.get("WASTECAP_QUANTIZE", "none")
        reviewer = os.environ.get("WASTECAP_REVIEWER", "annotator_1")
        db_path = os.environ.get("WASTECAP_DB_PATH", "data/db/wastecap.db")

        st.markdown(f":material/smart_toy: **{backend}** · `{device}`")
        st.caption(f"Quantization: `{quantize}` · Reviewer: `{reviewer}`")
        st.caption(f"Database: `{db_path}`")
        st.divider()

        st.caption("Use the pages above to move from generation to approval and export.")


def main() -> None:
    from app.components.ui import card, inject_styles, page_header, section_heading

    inject_styles()
    _sidebar()

    page_header(
        "WasteCap workspace",
        "Curate trustworthy waste intelligence",
        (
            "Generate candidates, ground every object, and approve only what a human "
            "reviewer can verify."
        ),
    )

    # Initialise DB and show status
    db_path = _init_db()

    from wastecap.store import count_records, source_counts, status_counts

    total = count_records(db_path=db_path)
    by_status = status_counts(db_path=db_path)
    by_source = source_counts(db_path=db_path)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total records", total)
    col2.metric("Approved", by_status.get("approved", 0))
    col3.metric("Pending", by_status.get("pending", 0))
    col4.metric("Rejected", by_status.get("rejected", 0))

    section_heading("Workspace overview", f"Source of truth · {db_path}")

    if by_source:
        section_heading("Collection mix", "Images currently in the source of truth")
        import pandas as pd

        df = pd.DataFrame(
            {"source": list(by_source.keys()), "count": list(by_source.values())}
        ).set_index("source")
        st.bar_chart(df, height=220)

    reviewed = by_status.get("approved", 0) + by_status.get("rejected", 0)
    section_heading("Review health", "Move the queue from pending to trusted")
    progress_col, summary_col = st.columns([1.5, 1])
    with progress_col, st.container(border=True):
        st.markdown("**Human review completion**")
        st.progress(reviewed / total if total else 0)
        st.caption(f"{reviewed} of {total} records have a final approve/reject decision.")
        st.metric(
            "Approval rate",
            f"{by_status.get('approved', 0) / total:.1%}" if total else "0.0%",
        )
    with summary_col, st.container(border=True):
        st.markdown("**Queue snapshot**")
        st.write(f":orange-badge[Pending] {by_status.get('pending', 0)}")
        st.write(f":green-badge[Approved] {by_status.get('approved', 0)}")
        st.write(f":red-badge[Rejected] {by_status.get('rejected', 0)}")

    section_heading("Workflow", "A simple path from image to validated export")
    a, b, c = st.columns(3)
    with a:
        card(
            "01 · Generate",
            "Curate",
            "Select an image, model, and prompt. Keep every candidate visible.",
        )
    with b:
        card("02 · Verify", "Review", "Edit, flag, and approve only what the image supports.")
    with c:
        card(
            "03 · Deliver",
            "Export",
            "Run schema validation before downloading split-level artifacts.",
        )

    if total == 0:
        st.warning(
            "No records in the database yet. "
            "Run `python scripts/ingest.py` to load images, "
            "or use the Curate page with the mock backend."
        )


if __name__ == "__main__":
    main()
