"""Export page — export approved annotations as JSON/JSONL/CSV."""

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_SRC_ROOT = str(Path(_PROJECT_ROOT) / "src")
for _path in (_PROJECT_ROOT, _SRC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.components.ui import inject_styles, page_header  # noqa: E402

from wastecap.export import ExportValidationError, export_splits  # noqa: E402

st.set_page_config(page_title="Export — WasteCap", page_icon="📤")
inject_styles()
page_header(
    "Delivery",
    "Export approved data",
    "Validate once, then download a clean artifact for each split.",
)

output_dir = Path(st.text_input("Output directory", "results/exports"))
split = st.selectbox("Preview split", ["train", "val", "test"], index=0)
include_all = st.checkbox(
    "Include non-approved records",
    value=False,
    help="Use only for diagnostics.",
)
st.caption("Exports are derived from the SQLite source of truth. Approved records are the default.")

if st.button("Validate and export all splits", type="primary", use_container_width=True):
    try:
        counts = export_splits(output_dir, include_all=include_all)
    except ExportValidationError as exc:
        st.error(str(exc))
    else:
        st.success(f"Validated and exported to `{output_dir}`.")
        st.json(counts)
        st.session_state["export_counts"] = counts

counts = st.session_state.get("export_counts")
if counts:
    st.subheader(f"{split.title()} downloads")
    row = st.columns(3)
    for column, extension, label, mime in zip(
        row,
        ("json", "jsonl", "csv"),
        ("JSON", "JSONL", "CSV"),
        ("application/json", "application/json", "text/csv"),
        strict=True,
    ):
        path = output_dir / f"{split}.{extension}"
        with column:
            if path.exists():
                st.download_button(
                    f"Download {label}",
                    path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    use_container_width=True,
                )
                st.caption(f"{counts[split][extension]} record(s)")
