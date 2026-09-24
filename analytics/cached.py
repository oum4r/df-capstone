"""Cached file reads for the pages: one st.cache_data wrapper per file.

load.py stays free of Streamlit so its builders can be tested on fixtures;
the caching lives here, in the page layer.
"""

import pandas as pd
import streamlit as st

import load

TRAINING_TARGETS = ["share", "ratio"]


@st.cache_data(show_spinner=False)
def json_file(name: str) -> dict:
    """A JSON file from DATA_DIR/processed, read once per session.

    :param name: file name, e.g. "metrics_day5.json".
    :return: the parsed dict.
    """
    return load.read_json(load.processed_path(name))


@st.cache_data(show_spinner=False)
def training() -> pd.DataFrame:
    """training.parquet: the 67,350 training rows with targets and features."""
    return load.read_parquet(load.processed_path("training.parquet"))


@st.cache_data(show_spinner=False)
def shelf_by_month() -> pd.DataFrame:
    """shelf_by_month.parquet: event counts per year, month and shelf."""
    return load.read_parquet(load.processed_path("shelf_by_month.parquet"))


@st.cache_data(show_spinner=False)
def oof_with_bucket(target: str, model: str) -> pd.DataFrame:
    """One target's out-of-fold predictions with each row's want-to-read bucket.

    :param target: "share" or "ratio".
    :param model: prediction column, e.g. "hist_gb".
    :return: columns work_key, y, model, bucket, in file order.
    """
    oof = load.read_parquet(load.processed_path(f"sweep_day5_{target}_oof.parquet"), columns=["work_key", "y", model])
    return load.attach_bucket(oof, training()[["work_key", "log1p_n_want"]])


@st.cache_data(show_spinner=False)
def oof_sample(target: str, model: str, n: int = 5000) -> pd.DataFrame:
    """A fixed-seed sample of one target's out-of-fold predictions.

    :param target: "share" or "ratio".
    :param model: prediction column, e.g. "hist_gb".
    :param n: rows to sample.
    :return: columns actual, predicted, bucket.
    """
    return load.oof_sample(oof_with_bucket(target, model), model, n=n)


@st.cache_data(show_spinner=False)
def spearman_all() -> pd.DataFrame:
    """Spearman of every stored feature with share and ratio, strongest first.

    :return: columns feature, share, ratio, ranked by the absolute value against share.
    """
    features = json_file("features.json")["feature_columns"]
    return load.spearman_table(training(), features, TRAINING_TARGETS)
