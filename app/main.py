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
from pathlib import Path

import streamlit as st

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
        st.title("♻️ WasteCap")
        st.caption("Waste Image–Caption Dataset Curation")
        st.divider()

        # Environment / config info
        st.subheader("Configuration")
        backend = os.environ.get("WASTECAP_BACKEND", "mock")
        device = os.environ.get("WASTECAP_DEVICE", "cpu")
        quantize = os.environ.get("WASTECAP_QUANTIZE", "none")
        reviewer = os.environ.get("WASTECAP_REVIEWER", "annotator_1")
        db_path = os.environ.get("WASTECAP_DB_PATH", "data/db/wastecap.db")

        st.markdown(f"**Backend:** `{backend}`")
        st.markdown(f"**Device:** `{device}`")
        st.markdown(f"**Quantize:** `{quantize}`")
        st.markdown(f"**Reviewer:** `{reviewer}`")
        st.markdown(f"**DB:** `{db_path}`")
        st.divider()

        st.caption("Navigate via the sidebar pages →")


def main() -> None:
    _sidebar()

    st.title("WasteCap — Dashboard")
    st.markdown(
        "Welcome to the **Waste Image–Caption Dataset Curation App**. "
        "Use the sidebar to navigate to individual pages."
    )
    st.divider()

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

    st.subheader("Database")
    st.info(f"DB path: `{db_path}`")

    if by_source:
        st.subheader("Records by source")
        st.bar_chart(by_source)

    if total == 0:
        st.warning(
            "No records in the database yet. "
            "Run `python scripts/ingest.py` to load images, "
            "or use the Curate page with the mock backend."
        )


if __name__ == "__main__":
    main()
