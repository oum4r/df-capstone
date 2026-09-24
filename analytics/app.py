"""Epilogue analytics: the project's data, models and evaluations in one place.

Streamlit reads the theme (and static serving for the font) from
.streamlit/config.toml in the folder it is started from.

Figures are read at run time from DATA_DIR (CAPSTONE_DATA, or the bundled
analytics/data); fixed text lives in content.py.
"""

import streamlit as st

import theme

st.set_page_config(page_title="Epilogue analytics", layout="wide")
theme.inject_css()

PAGES = [
    st.Page("pages/overview.py", title="Overview", default=True),
    st.Page("pages/data.py", title="Data"),
    st.Page("pages/models.py", title="Models"),
    st.Page("pages/features.py", title="Features"),
    st.Page("pages/llm_evaluation.py", title="LLM evaluation"),
    st.Page("pages/stack.py", title="Stack"),
]

st.navigation(PAGES, position="top").run()
