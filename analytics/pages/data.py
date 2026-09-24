"""Data: datasets, the filter funnel, the target, gaps, correlations, seasonality."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import cached
import content
import load
import theme
import names
import ui

TARGET_UPPER = {"share": 1.0, "ratio": 3.0}
TOP_CORRELATIONS = 12
BOTH_SHELVES = "Want to Read and Already Read"
DEFAULT_SHELVES = ["Want to Read", "Already Read"]
# Too few events for a monthly pattern.
SPARSE_SHELVES = ["Stopped Reading"]

notes = ui.Notes()

st.title("Data")
st.markdown(
    "Where the data comes from, how the training books were chosen, and what share, the figure the "
    "model predicts, looks like."
)

st.header("Datasets")
ui.static_table(
    pd.DataFrame(content.DATASETS),
    labels={"dataset": "Dataset", "what": "What it is", "size": "Size", "became": "What it became",
            "status": "Status"},
)

st.header("From every shelved book to the training books")
metrics = cached.json_file("metrics_day5.json")
training = cached.training()
funnel = pd.DataFrame(content.FUNNEL)
funnel["share_of_all"] = 100 * funnel["works"] / funnel["works"].iloc[0]
left, right = st.columns([3, 2], gap="large")
with left:
    ui.static_table(
        funnel,
        order=["stage", "works", "share_of_all"],
        labels={"stage": "Stage", "works": "Books", "share_of_all": "Of all shelved"},
        formats={"works": "{:,}", "share_of_all": "{:.0f}%"},
        numeric=["works", "share_of_all"],
    )
    check = "match" if metrics["share"]["n_rows"] == len(training) == funnel["works"].iloc[-1] else "do not match"
with right:
    st.markdown(
        "A book is kept only with at least 5 Want to Read and 3 Already Read shelvings, which keeps "
        f"about {funnel['share_of_all'].iloc[1]:.0f}% of all shelved books."
    )

st.header("How share is spread")
st.markdown(
    "Share lies between 0 and 1 and is the figure the app's model predicts. Ratio, already read / want to "
    "read, is the older target, kept for comparison. It has no upper limit, and a few books reach "
    f"{int(training['ratio'].max())}."
)
c1, c2, _ = st.columns([1, 1, 2], gap="large")
with c1:
    target = st.selectbox("Target", list(TARGET_UPPER), format_func=names.TARGETS.get, index=0, key="data_target")
with c2:
    bins = st.slider("Number of bars", min_value=10, max_value=100, value=40, step=5, key="data_bins")
upper = TARGET_UPPER[target]
values = training[target]


def histogram(values: pd.Series, bins: int, upper: float, title: str) -> alt.Chart:
    """Bar histogram with a dashed median rule, both with tooltips.

    :param values: the column.
    :param bins: bin count.
    :param upper: right edge.
    :param title: x-axis title.
    :return: the chart.
    """
    hist = load.histogram_table(values, bins, upper)
    bars = alt.Chart(hist).mark_bar(color=theme.NEUTRAL_FILL).encode(
        x=alt.X("start:Q", title=title, bin="binned", scale=alt.Scale(domain=[0, upper])),
        x2="end:Q",
        y=alt.Y("count:Q", title="Books"),
        tooltip=[alt.Tooltip("start:Q", title="From", format=".3f"), alt.Tooltip("end:Q", title="To", format=".3f"),
                 alt.Tooltip("count:Q", title="Books", format=",")],
    )
    median = alt.Chart(pd.DataFrame({"median": [values.median()]})).mark_rule(
        color=theme.INK, strokeDash=[4, 3], strokeWidth=1.5
    ).encode(x="median:Q", tooltip=[alt.Tooltip("median:Q", title="Median", format=".3f")])
    return (bars + median).properties(height=320)


def summary_value(measure: str, value: float) -> str:
    """Counts with separators, the maximum to 2 decimals, the rest to 3.

    :param measure: the summary row's measure.
    :param value: its figure.
    :return: its display text.
    """
    if measure in ("Works", "Books") or measure.startswith("Above"):
        return f"{value:,.0f}"
    return f"{value:.2f}" if measure == "Maximum" else f"{value:.3f}"


left, right = st.columns([3, 2], gap="large")
with left:
    theme.show_chart(histogram(values, bins, upper, names.TARGET_SHORT[target]))
with right:
    summary = load.distribution_summary(values, upper if target == "ratio" else None)
    if target == "share":
        summary = summary[summary["measure"] != "Maximum"]
    summary = summary.assign(
        shown=[summary_value(m, v) for m, v in zip(summary["measure"], summary["value"])],
        measure=summary["measure"].replace({"Works": "Books"}),
    )
    ui.static_table(
        summary,
        order=["measure", "shown"],
        labels={"measure": "Measure", "shown": names.TARGET_SHORT[target]},
        numeric=["shown"],
    )

st.header("Missing values")
features = cached.json_file("features.json")["feature_columns"]
missing = load.missing_table(training[["share", "ratio", *features]])
missing["share_missing"] = 100 * missing["share_missing"]
missing["name"] = [names.input_name(c) for c in missing["column"]]
n_complete = len(features) + 2 - len(missing)
n_flags = sum(c.startswith("subj_") for c in features)
n_topics = len(cached.json_file("explain_day5.json")["svd_labels"])
st.markdown(
    f"Of the {len(features) + 2} stored columns (2 targets and {len(features)} candidate inputs, of which the "
    f"model uses {len(features) - n_flags} directly and turns the {n_flags} subject flags into {n_topics} topics), "
    f"{n_complete} are complete and {len(missing)} have gaps. {missing['name'].iloc[0]} is missing most often, "
    f"for {missing['share_missing'].iloc[0]:.1f}% of books."
    if len(missing) else "No stored column has a gap."
)
left, right = st.columns(2, gap="large")
with left:
    ui.static_table(
        missing,
        order=["name", "n_missing", "share_missing"],
        labels={"name": "Column", "n_missing": "Books missing it", "share_missing": "Of all books"},
        formats={"n_missing": "{:,}", "share_missing": "{:.1f}%"},
        numeric=["n_missing", "share_missing"],
    )
with right:
    gaps = alt.Chart(missing).mark_bar(color=theme.NEUTRAL_FILL).encode(
        x=alt.X("share_missing:Q", title="Books missing it (%)"),
        y=alt.Y("name:N", sort=list(missing["name"]), title=None),
        tooltip=[alt.Tooltip("name:N", title="Column"), alt.Tooltip("n_missing:Q", title="Books missing it", format=","),
                 alt.Tooltip("share_missing:Q", title="Of all books (%)", format=".1f")],
    ).properties(height=44 * max(len(missing), 1))
    theme.show_chart(gaps)

st.header("Which inputs move with share")
ui.explain("spearman")
corr_all = cached.spearman_all()
same = bool(np.allclose(corr_all["share"], corr_all["ratio"], equal_nan=True))
picked = st.multiselect(
    "Inputs to show",
    list(corr_all["feature"]),
    default=list(corr_all["feature"].head(TOP_CORRELATIONS)),
    format_func=names.input_name,
    key="data_features",
    help="Listed strongest first. The default is the top 12.",
)
corr = load.filter_rows(corr_all, "feature", picked)
corr["name"] = [names.input_name(f) for f in corr["feature"]]
top = corr_all["feature"].iloc[0]
reading = (
    ": books with more Want to Read shelvings have a lower share, partly by construction, since that count "
    "is in share's denominator."
    if top == "log1p_n_want" else "."
)
st.markdown(
    f"How closely each of the {len(corr_all)} stored inputs moves with share, by rank. The strongest is "
    f"{names.input_name(top)}, at {corr_all['share'].iloc[0]:.3f}{reading}"
    + (
        " Share and ratio give the same figures: share = ratio / (1 + ratio) keeps every book in the "
        "same rank order, and Spearman reads only the order."
        if same else ""
    )
)
if corr.empty:
    st.markdown("Choose at least one input to see its correlation.")
else:
    label = "Spearman with share" + (" and ratio" if same else "")
    left, right = st.columns(2, gap="large")
    with left:
        rho = alt.Chart(corr).mark_bar(color=theme.NEUTRAL_FILL).encode(
            x=alt.X("share:Q", title=label),
            y=alt.Y("name:N", sort=list(corr["name"]), title=None),
            tooltip=[alt.Tooltip("name:N", title="Input"), alt.Tooltip("share:Q", title="With share", format=".3f"),
                     alt.Tooltip("ratio:Q", title="With ratio", format=".3f")],
        ).properties(height=30 * len(corr) + 20)
        theme.show_chart(rho)
    with right:
        ui.static_table(
            corr,
            order=["name", "share"] if same else ["name", "share", "ratio"],
            labels={"name": "Input", "share": label, "ratio": "Spearman with ratio"},
            formats={"share": "{:.3f}", "ratio": "{:.3f}"},
            numeric=["share", "ratio"],
        )

st.header("Seasonality")
months = cached.shelf_by_month()
years = load.complete_years(months)
shelves = [s for s in dict.fromkeys(months["shelf"]) if s not in SPARSE_SHELVES]
c1, _ = st.columns([1, 2], gap="large")
with c1:
    choice = st.selectbox("Shelf", [BOTH_SHELVES, *shelves], index=0, key="data_shelf")
chosen = DEFAULT_SHELVES if choice == BOTH_SHELVES else [choice]
profile = load.seasonal_profile(months, chosen, years)
month_order = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
profile["month_name"] = [month_order[m - 1] for m in profile["month"]]
st.markdown(
    f"For each full year, each month's shelvings as a percentage of that year's total, averaged over "
    f"{years.start} to {years.stop - 1}. Working in percentages per year stops the growth of Open Library's "
    "logs from favouring recent months."
)
in_years = months[months["year"].isin(set(years))]
for shelf in SPARSE_SHELVES:
    per_year = in_years[in_years["shelf"] == shelf].groupby("year")["count"].sum()
    empty = [str(y) for y, n in per_year.items() if n == 0]
    notes.add(
        "Seasonality",
        f"{shelf} is not offered: it has only {int(per_year.sum()):,} events in {years.start} to {years.stop - 1}"
        + (f" (none in {', '.join(empty)})" if empty else "")
        + ", too few for a monthly pattern.",
    )
left, right = st.columns([3, 2], gap="large")
with left:
    season = alt.Chart(profile).mark_line(point=True).encode(
        x=alt.X("month_name:N", title="Month", sort=month_order),
        y=alt.Y("share:Q", title="Percent of the year's shelvings", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "shelf:N",
            title="Shelf",
            scale=alt.Scale(domain=chosen, range=theme.series_colours(chosen, None)),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[alt.Tooltip("shelf:N", title="Shelf"), alt.Tooltip("month_name:N", title="Month"),
                 alt.Tooltip("share:Q", title="Percent of the year's shelvings", format=".2%")],
    ).properties(height=340)
    theme.show_chart(season)
with right:
    wide = profile.pivot(index="month", columns="shelf", values="share").reindex(columns=chosen).reset_index()
    wide["month"] = [month_order[m - 1] for m in wide["month"]]
    ui.static_table(
        wide,
        labels={"month": "Month"},
        formats={s: "{:.2%}" for s in chosen},
        numeric=chosen,
    )

notes.render()
