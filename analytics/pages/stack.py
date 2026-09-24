"""Stack: the external APIs, the language models and the regression models."""

import pandas as pd
import streamlit as st

import cached
import content
import load
import ui

# The app's model settings in plain words: parameter -> (label, value to text).
SETTINGS = {
    "max_iter": ("boosting rounds", lambda v: f"{v}"),
    "max_depth": ("depth limit", lambda v: "none" if v is None else f"{v}"),
    "learning_rate": ("learning rate", lambda v: f"{v}"),
    "l2_regularization": ("L2 regularisation", lambda v: f"{v}"),
}


def setting_text(key: str, value: object) -> str:
    """One model setting in plain words, e.g. "200 boosting rounds" or "no depth limit".

    :param key: the parameter name.
    :param value: its value.
    :return: the phrase.
    """
    label, show = SETTINGS.get(key, (key, str))
    if key == "max_depth" and value is None:
        return "no depth limit"
    if key == "max_iter":
        return f"{show(value)} {label}"
    return f"{label} {show(value)}"


st.title("Stack")
st.markdown(
    "The services and models the app calls, each checked against the code that calls it. The offline "
    "summary judges and the model kept for comparison, which the app does not use, are listed too."
)

st.header("APIs")
ui.static_table(
    pd.DataFrame(content.APIS),
    labels={"api": "API", "endpoints": "Endpoints", "use": "Used for"},
    wrap_anywhere=["endpoints"],
)

st.header("Language models")
ui.static_table(
    pd.DataFrame(content.LLMS),
    labels={"model": "Model", "provider": "Provider", "role": "Role"},
)

st.header("Regression models")
params = cached.json_file("metrics_day5.json")[load.SHIPPED_TARGET]["families"][load.SHIPPED_FAMILY]["params"]
ui.static_table(pd.DataFrame(content.REGRESSORS), labels={"model": "Model", "role": "Role"})
st.markdown("Settings in the app: " + ", ".join(setting_text(k, v) for k, v in params.items()) + ".")
