"""Shared visual language for the WasteCap Streamlit pages."""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --wc-ink:#15202b; --wc-muted:#667482; --wc-teal:#0b8f82;
                --wc-teal-dark:#08756b; --wc-mint:#dcf7f1; --wc-border:#e4eaee;
                --wc-bg:#f4f7f8; --wc-navy:#12232e; --wc-orange:#e8894a; }
        .stApp { background:var(--wc-bg); }
        [data-testid="stAppViewContainer"] > .main { background:var(--wc-bg); }
        .block-container { max-width:1420px; padding-top:2.1rem; padding-bottom:4rem; }
        .stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stCaptionContainer"] {
          color:var(--wc-ink);
        }
        .stApp [data-testid="stCaptionContainer"], .stApp small { color:var(--wc-muted); }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] {
          color:var(--wc-ink); background:#ffffff;
        }
        .stApp input::placeholder, .stApp textarea::placeholder { color:#71808a; opacity:1; }
        [data-testid="stHeader"] { background:rgba(244,247,248,.92); }
        [data-testid="stSidebar"] { border-right:1px solid #203845; }
        [data-testid="stSidebar"] > div:first-child { background:var(--wc-navy); }
        [data-testid="stSidebar"] * { color:#e7f2f1; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#9fb5b8; }
        [data-testid="stSidebar"] hr { border-color:#2b4650; }
        [data-testid="stSidebar"] [data-baseweb="select"],
        [data-testid="stSidebar"] input { background:#203943; border-color:#385661; }
        .wc-brand { display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.4rem; }
        .wc-brand-mark { display:grid; place-items:center; width:2.25rem; height:2.25rem;
          border-radius:10px; color:#073f3a; background:#71e1cf; font-weight:900; }
        .wc-brand-name { color:#fff; font-size:1.15rem; font-weight:800; letter-spacing:-.02em; }
        .wc-brand-sub { color:#9fb5b8; font-size:.7rem; margin-top:.1rem; }
        .wc-hero { position:relative; overflow:hidden; padding:2rem 2.2rem;
          border:1px solid #dce8e7;
          border-radius:24px;
          background:linear-gradient(115deg,#ffffff 0%,#e7f8f4 62%,#d1f0eb 100%);
          margin-bottom:1.5rem; box-shadow:0 12px 30px rgba(20,54,60,.06); }
        .wc-hero:after { content:""; position:absolute; width:240px; height:240px;
          right:-70px; top:-100px;
          border:34px solid rgba(11,143,130,.1); border-radius:50%; }
        .wc-eyebrow { color:var(--wc-teal-dark); font-size:.7rem; font-weight:800;
          letter-spacing:.14em; text-transform:uppercase; }
        .wc-hero h1 { color:var(--wc-ink); margin:.35rem 0 .55rem; font-size:2.25rem;
          line-height:1.08; letter-spacing:-.045em; }
        .wc-hero p { color:var(--wc-muted); margin:0; max-width:680px; font-size:1rem; }
        .wc-card { background:#fff; border:1px solid var(--wc-border);
          border-radius:18px; padding:1.15rem 1.25rem; height:100%;
          box-shadow:0 5px 16px rgba(30,52,61,.045); }
        .wc-kicker { color:var(--wc-muted); font-size:.78rem; text-transform:uppercase;
          letter-spacing:.06em; font-weight:700; }
        .wc-value { color:var(--wc-ink); font-size:1.85rem; font-weight:800;
          margin-top:.2rem; letter-spacing:-.04em; }
        .wc-muted { color:var(--wc-muted); }
        .wc-section { display:flex; align-items:center; justify-content:space-between;
          margin:1.8rem 0 .7rem; }
        .wc-section h2 { margin:0; color:var(--wc-ink); font-size:1.1rem;
          letter-spacing:-.02em; }
        .wc-section span { color:var(--wc-muted); font-size:.82rem; }
        div[data-testid="stMetric"] { background:#fff; border:1px solid var(--wc-border);
          padding:1.05rem 1.15rem; border-radius:18px; box-shadow:0 5px 16px rgba(30,52,61,.045); }
        div[data-testid="stMetricLabel"] { color:var(--wc-muted); }
        div[data-testid="stMetricValue"] { color:var(--wc-ink); font-weight:800;
          letter-spacing:-.04em; }
        div.stButton > button { border-radius:11px; font-weight:700; min-height:2.55rem; }
        div[data-testid="stExpander"] { border-color:var(--wc-border);
          border-radius:16px; background:#fff; }
        div[data-testid="stProgress"] > div > div { background:var(--wc-teal); }
        [data-testid="stDataFrame"] { border:1px solid var(--wc-border);
          border-radius:14px; overflow:hidden; }
        .wc-status { display:inline-flex; align-items:center; gap:.35rem; padding:.3rem .6rem;
          border-radius:99px; background:var(--wc-mint); color:var(--wc-teal-dark);
          font-size:.75rem; font-weight:750; }
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


def section_heading(title: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="wc-section"><h2>{title}</h2><span>{detail}</span></div>',
        unsafe_allow_html=True,
    )


def card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="wc-card"><div class="wc-kicker">{label}</div>'
        f'<div class="wc-value">{value}</div><div class="wc-muted">{detail}</div></div>',
        unsafe_allow_html=True,
    )
