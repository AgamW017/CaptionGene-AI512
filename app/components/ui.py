"""Shared visual language for the WasteCap Streamlit pages."""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --wc-ink:#17212b; --wc-muted:#657786; --wc-teal:#0f766e;
                --wc-mint:#dff7f2; --wc-border:#dfe7ea; --wc-bg:#f7fafb; }
        .stApp { background:var(--wc-bg); }
        [data-testid="stHeader"] { background:rgba(247,250,251,.9); }
        [data-testid="stSidebar"] { border-right:1px solid var(--wc-border); }
        [data-testid="stSidebar"] > div:first-child { background:#ffffff; }
        .wc-hero { padding:1.25rem 1.4rem; border:1px solid var(--wc-border);
          border-radius:18px; background:linear-gradient(135deg,#ffffff 0%,#eafaf7 100%);
          margin-bottom:1.2rem; }
        .wc-eyebrow { color:var(--wc-teal); font-size:.75rem; font-weight:700;
          letter-spacing:.12em; text-transform:uppercase; }
        .wc-hero h1 { color:var(--wc-ink); margin:.25rem 0 .4rem; font-size:2rem; }
        .wc-hero p { color:var(--wc-muted); margin:0; }
        .wc-card { background:#fff; border:1px solid var(--wc-border);
          border-radius:14px; padding:1rem 1.1rem; height:100%; }
        .wc-kicker { color:var(--wc-muted); font-size:.78rem; text-transform:uppercase;
          letter-spacing:.06em; font-weight:700; }
        .wc-value { color:var(--wc-ink); font-size:1.7rem; font-weight:750; margin-top:.15rem; }
        .wc-muted { color:var(--wc-muted); }
        div[data-testid="stMetric"] { background:#fff; border:1px solid var(--wc-border);
          padding:1rem; border-radius:14px; }
        div.stButton > button { border-radius:10px; font-weight:650; }
        div[data-testid="stExpander"] { border-color:var(--wc-border); border-radius:14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<section class="wc-hero"><div class="wc-eyebrow">{eyebrow}</div>'
        f"<h1>{title}</h1><p>{description}</p></section>",
        unsafe_allow_html=True,
    )


def card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="wc-card"><div class="wc-kicker">{label}</div>'
        f'<div class="wc-value">{value}</div><div class="wc-muted">{detail}</div></div>',
        unsafe_allow_html=True,
    )
