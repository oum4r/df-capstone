"""Features: the model's inputs, how they are prepared before training, and which ones matter most."""

import altair as alt
import pandas as pd
import streamlit as st

import cached
import content
import load
import names
import theme
import ui

TARGETS = ["share", "ratio"]
ALL_GROUPS = "All groups"
METHODS = {
    "permutation": ("Permutation importance", "MAE increase"),
    "shap_mean_abs": ("Mean absolute SHAP (size of contribution)", "Mean absolute SHAP (log-odds scale)"),
}

stored = cached.json_file("features.json")["feature_columns"]
explain = cached.json_file("explain_day5.json")
metrics = cached.json_file("metrics_day5.json")
svd_labels = explain["svd_labels"]

features = load.model_features(stored, svd_labels)
n_svd = sum(f.startswith("svd_") for f in features)
n_subj = sum(c.startswith("subj_") for c in stored)
n_facts = len(features) - n_svd

st.title("Features")
st.markdown("What the model reads about each book, how those inputs are prepared, and which ones matter most.")

st.header("The model's inputs")
groups = load.feature_groups_table(features, content.FEATURE_GROUPS, svd_labels)
groups["name"] = [names.input_name(f) for f in groups["feature"]]
c1, _ = st.columns([1, 2], gap="large")
with c1:
    group = st.selectbox("Input group", [ALL_GROUPS, *dict.fromkeys(groups["group"])], index=0, key="features_group")
shown = groups if group == ALL_GROUPS else load.filter_rows(groups, "group", [group])
left, right = st.columns([3, 2], gap="large")
with left:
    ui.static_table(shown, order=["group", "name"], labels={"group": "Group", "name": "Input"})
with right:
    st.markdown(
        f"The app's model uses {len(features)} inputs: {n_facts} facts about each book and {n_svd} subject "
        "topics. The topics are built from all of a book's Open Library subject words, weighting rare words more "
        f"(TF-IDF) and reducing them to {n_svd} topics (truncated SVD); they replace the {n_subj} yes-or-no subject "
        "flags stored beside them. Each topic is labelled by its strongest words; a word after \"not\" pulls the "
        "topic the other way (a minus sign in the stored labels)."
    )
    st.markdown(
        f"The Data page counts {len(stored) + 2} stored columns: the 2 targets and {len(stored)} candidate inputs, "
        f"which are these {n_facts} facts plus the {n_subj} subject flags."
    )
    missed = load.ungrouped(features, groups)
    explained = {e["name"] for e in explain["permutation"]}
    if missed or explained != set(features):
        st.markdown(
            f"Check: {len(missed)} inputs are outside the grouping, and the inputs "
            f"{'do not match' if explained != set(features) else 'match'} the importance file."
        )
    if group != ALL_GROUPS:
        st.markdown(f"Showing {len(shown)} of the {len(groups)} inputs.")

st.header("How the inputs are prepared")
st.markdown(
    "Before training, the model search tried several ways to prepare the inputs and kept the best for each "
    "model. These are the tuned HistGB's choices on each target."
)
arms = load.arm_table(metrics, load.SHIPPED_FAMILY, TARGETS)


def arm_notes(target: str) -> pd.DataFrame:
    """The preparation for one target, in display names, with what each choice does.

    :param target: "share" or "ratio".
    :return: columns setting, value, meaning.
    """
    rows = []
    for setting, value in zip(arms["setting"], arms[target]):
        meaning = content.ARM_NOTES.get((setting, value), "")
        rows.append({"setting": names.arm_setting(setting), "value": names.arm_value(setting, value),
                     "meaning": meaning})
    return pd.DataFrame(rows)


for column, target in zip(st.columns(2, gap="large"), TARGETS):
    with column:
        st.subheader(names.TARGETS[target])
        ui.static_table(
            arm_notes(target),
            labels={"setting": "Setting", "value": "Choice", "meaning": "What it does"},
        )

st.header("What drives the prediction")
st.markdown(
    "Two ways to rank the inputs of the app's model, HistGB on share. Permutation importance shuffles one input "
    "at a time and measures how much the MAE rises on one test fold of unseen authors, on the share scale. Mean "
    "absolute SHAP is the average size of each input's contribution to a prediction, on the model's log-odds "
    "scale. The two scales differ, so compare the order, not the sizes."
)
c1, c2, _ = st.columns([1, 1, 1], gap="large")
with c1:
    method = st.selectbox("Method", list(METHODS), format_func=lambda m: METHODS[m][0], index=0, key="features_method")
with c2:
    top = st.slider("Inputs shown", min_value=5, max_value=len(features), value=15, step=1, key="features_top")
drivers = load.importance_table(explain[method], svd_labels, top)
drivers["name"] = [names.input_name(f) for f in drivers["feature"]]
title, axis_title = METHODS[method]
method_note = explain["method"]["repeats"] if method == "permutation" else explain["method"]["sample_size"]
left, right = st.columns([3, 2], gap="large")
with left:
    chart = alt.Chart(drivers).mark_bar(color=theme.NEUTRAL_FILL).encode(
        x=alt.X("mean:Q", title=axis_title),
        y=alt.Y("name:N", sort=list(drivers["name"]), title=None),
        tooltip=[alt.Tooltip("name:N", title="Input"), alt.Tooltip("mean:Q", title="Average", format=".4f"),
                 alt.Tooltip("std:Q", title="Spread (std)", format=".4f")],
    ).properties(height=28 * len(drivers) + 20)
    theme.show_chart(chart)
with right:
    ui.static_table(
        drivers,
        order=["name", "mean", "std"],
        labels={"name": "Input", "mean": "Average", "std": "Spread (std)"},
        formats={"mean": "{:.4f}", "std": "{:.4f}"},
        numeric=["mean", "std"],
    )
