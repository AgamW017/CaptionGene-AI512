"""
Caption card component.

Renders one Candidate as a card showing:
  - The caption text (or parse_failed warning)
  - CLIPScore badge (grey if not yet computed)
  - Model + prompt_id tag
  - Approve / Reject / Edit buttons

Returns the action taken by the user: "approve", "reject", "edit", or None.

Usage (in a page)
-----------------
    action = caption_card(candidate, key_prefix="cand_0")
    if action == "approve":
        ...
"""

from __future__ import annotations

import streamlit as st

from wastecap.schema import Candidate


def caption_card(
    candidate: Candidate,
    *,
    key_prefix: str = "",
    show_buttons: bool = True,
    selected: bool = False,
) -> str | None:
    """Render a caption candidate card.

    Args:
        candidate:    The Candidate to display.
        key_prefix:   Unique prefix for Streamlit widget keys (avoid key collisions
                      when multiple cards are rendered in a loop).
        show_buttons: If False, renders read-only (used in Compare page).
        selected:     Highlight the card if True (used when this candidate is
                      the currently active edit target).

    Returns:
        One of "approve", "reject", "edit", or None if no button was clicked.
    """
    with st.container(border=True):
        # ── Header row: model tag + CLIPScore badge ──
        col_model, col_score = st.columns([3, 1])
        with col_model:
            st.caption(f"🤖 `{candidate.model}` · `{candidate.prompt_id}`")
        with col_score:
            if candidate.clip_score is not None:
                score_val = candidate.clip_score
                colour = "green" if score_val >= 0.28 else "orange" if score_val >= 0.22 else "red"
                st.markdown(
                    f"**CLIPScore** :{colour}[{score_val:.3f}]",
                    help="CLIPScore ≥ 0.28 = good, 0.22–0.28 = marginal, <0.22 = low",
                )
            else:
                st.caption("CLIPScore: —")

        # ── Parse-failure warning ──
        if candidate.parse_failed:
            st.warning(
                "Structured fields unavailable — raw model text shown. Review before approving.",
                icon="⚠️",
            )

        # ── Caption text ──
        st.markdown(f"> {candidate.text[:500]}" if candidate.text else "> *(empty)*")

        if not show_buttons:
            return None

        # ── Action buttons ──
        b_approve, b_reject, b_edit = st.columns(3)
        action: str | None = None

        with b_approve:
            if st.button(
                "✅ Approve",
                key=f"{key_prefix}_approve",
                use_container_width=True,
                type="primary",
            ):
                action = "approve"
        with b_reject:
            if st.button(
                "❌ Reject",
                key=f"{key_prefix}_reject",
                use_container_width=True,
            ):
                action = "reject"
        with b_edit:
            if st.button(
                "✏️ Edit",
                key=f"{key_prefix}_edit",
                use_container_width=True,
            ):
                action = "edit"

    return action
