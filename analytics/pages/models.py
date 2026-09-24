"""Models: every model on both targets, why the app uses HistGB, where it errs, then background checks."""

import altair as alt
import pandas as pd
import streamlit as st

import cached
import content
import load
import names
import theme
import ui

MEASURES = {"mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R squared", "spearman": "Spearman",
            "mae_std": "Spread across folds"}
CHART_MEASURES = ["mae", "mse", "rmse", "r2", "spearman"]
DEFAULT_FAMILIES = [load.SHIPPED_FAMILY, load.DEFAULT_FAMILY, "catboost", "xgboost", load.BASELINE_FAMILY]
RATIO_SCATTER_MAX = 3.0
NO_COLOUR = "None"
BY_BUCKET = "Want to Read count"
REFERENCE = "reference"
HISTORY_LABELS = {"CatBoost": "Day 5 CatBoost (most accurate, not used)"}
TOGGLED = "Changes the families table, the bar chart, predicted against actual and error by Want to Read count. The other sections are fixed."

metrics = cached.json_file("metrics_day5.json")
notes = ui.Notes()


def gain_text(value: float) -> str:
    """A gain to 4 decimals, printing a tiny negative as 0.0000 rather than -0.0000.

    :param value: the gain.
    :return: its display text.
    """
    text = f"{value:.4f}"
    return "0.0000" if text == "-0.0000" else text


st.title("Models")
share_families = metrics["share"]["families"]
n_searched = sum(f.get("search_mae_3fold") is not None for f in share_families.values())
st.markdown(
    f"{len(share_families)} models were scored on both targets, share (the one the app uses) and the older ratio: "
    f"{n_searched} model types, plus HistGB with its default settings as a control. Each model was tested "
    f"{metrics['share']['n_splits']} times, each time on books by authors it had not seen in training, so an "
    "author's other books never help predict their own."
)

c1, _ = st.columns([1, 2], gap="large")
with c1:
    target = st.segmented_control(
        "Target for the model tables and charts", list(names.TARGETS), default="share",
        format_func=names.TARGETS.get, required=True, key="models_target", help=TOGGLED,
    ) or "share"
block = metrics[target]
shipped_name = load.SHIPPED_FAMILY if target == load.SHIPPED_TARGET else None


def family(key: str) -> str:
    """Display name of a model family for the chosen target."""
    return names.family_name(key, target)


st.header("All models")
ui.explain("mae", "mse", "rmse", "r2", "spearman", "fold_std")
families = load.families_table(metrics, target)
families["name"] = [family(f) for f in families["family"]]
chosen = st.multiselect(
    "Models shown in this table and in Error by Want to Read count",
    list(families["family"]),
    default=[f for f in DEFAULT_FAMILIES if f in set(families["family"])],
    format_func=family,
    key="models_families",
)
st.markdown(
    "Sorted by MAE."
    + (" The app's model is set in red." if target == load.SHIPPED_TARGET else
       " The app does not use a ratio model; its model is the tuned HistGB on share.")
)
table = load.filter_rows(families, "family", chosen)
if table.empty:
    st.markdown("Choose at least one model to fill the table.")
else:
    ui.static_table(
        table,
        order=["name", "note", *MEASURES],
        labels={"name": "Model", "note": "Note", **MEASURES},
        formats={c: "{:.4f}" for c in MEASURES},
        numeric=list(MEASURES),
        highlight=list(table["shipped"]),
    )
notes.add(
    "All models",
    "MSE here is RMSE squared, which is exact: the stored RMSE is the square root of the MSE pooled over "
    "the five test folds.",
)

c1, _ = st.columns([1, 2], gap="large")
with c1:
    measure = st.selectbox(
        "Measure in the chart", CHART_MEASURES, format_func=MEASURES.get, index=0, key="models_measure"
    )
ranked = load.rank_by_metric(families, measure).dropna(subset=[measure])
better = "higher" if measure in load.HIGHER_IS_BETTER else "lower"
bars = alt.Chart(ranked).mark_bar().encode(
    x=alt.X(f"{measure}:Q", title=f"{MEASURES[measure]} ({better} is better)"),
    y=alt.Y("name:N", sort=list(ranked["name"]), title=None),
    color=alt.condition(alt.datum.shipped, alt.value(theme.ACCENT), alt.value(theme.NEUTRAL_FILL)),
    tooltip=[alt.Tooltip("name:N", title="Model"), alt.Tooltip("note:N", title="Note"),
             alt.Tooltip(f"{measure}:Q", title=MEASURES[measure], format=".4f")],
).properties(height=26 * len(ranked))
theme.show_chart(bars)
dropped = len(families) - len(ranked)

st.header("Why the app uses the tuned HistGB (share)")
ship = load.ship_comparison(metrics)
rivals = ship[~ship["shipped"]].set_index("family")
st.markdown(
    f"CatBoost and XGBoost score a little better on share, by {-rivals.loc['catboost', 'gap']:.4f} and "
    f"{-rivals.loc['xgboost', 'gap']:.4f} MAE, or {-rivals.loc['catboost', 'gap'] * 100:.2f} and "
    f"{-rivals.loc['xgboost', 'gap'] * 100:.2f} percentage points on the share a reader sees. The gain is steady "
    "across folds but too small to matter, and HistGB is the simpler model: it comes with scikit-learn, which the "
    f"app already installs, and fits in seconds. Before the model search, the plan set {load.SHIP_MARGIN:.4f} MAE "
    "as the margin an ensemble had to beat the best single model by, and the project later required the same "
    "margin of every accuracy change. Both gaps here are well under it."
)
ship["name"] = [names.family_name(f, load.SHIPPED_TARGET) for f in ship["family"]]
ship["gap_text"] = [REFERENCE if base else f"{g:+.4f}" for base, g in zip(ship["shipped"], ship["gap"])]
ship["folds_text"] = [REFERENCE if base else f"{f:.0f}" for base, f in zip(ship["shipped"], ship["folds_lower"])]
ui.static_table(
    ship,
    order=["name", "mae", "gap_text", "folds_text", "fit_seconds", "extra_step"],
    labels={
        "name": "Model", "mae": "MAE", "gap_text": "MAE difference from HistGB (negative is more accurate)",
        "folds_text": "Test folds where it beat HistGB (of 5)", "fit_seconds": "Fit time (seconds)",
        "extra_step": "Extra input preparation at prediction time",
    },
    formats={"mae": "{:.4f}", "fit_seconds": "{:.1f}"},
    numeric=["mae", "gap_text", "folds_text", "fit_seconds"],
    highlight=list(ship["shipped"]),
)

st.header("Predicted against actual")
c1, c2, _ = st.columns([1, 1, 1], gap="large")
with c1:
    n_points = st.slider(
        "Points in the sample", min_value=1000, max_value=20000, value=5000, step=1000, key="models_points"
    )
with c2:
    colour_by = st.selectbox("Colour by", [NO_COLOUR, BY_BUCKET], index=0, key="models_colour")
sample = cached.oof_sample(target, load.SHIPPED_FAMILY, n_points)
short = names.TARGET_SHORT[target]
upper = 1.0 if target == "share" else RATIO_SCATTER_MAX
outside = int(((sample["actual"] > upper) | (sample["predicted"] > upper)).sum())
shown = sample[(sample["actual"] <= upper) & (sample["predicted"] <= upper)]
tips = [alt.Tooltip("actual:Q", title=f"Actual {short.lower()}", format=".3f"),
        alt.Tooltip("predicted:Q", title=f"Predicted {short.lower()}", format=".3f"),
        alt.Tooltip("bucket:N", title="Want to Read count")]
points = alt.Chart(shown).mark_circle(size=14, opacity=0.35, clip=True).encode(
    x=alt.X("actual:Q", title=f"Actual {short.lower()}", scale=alt.Scale(domain=[0, upper])),
    y=alt.Y("predicted:Q", title=f"Predicted {short.lower()}", scale=alt.Scale(domain=[0, upper])),
    color=(
        alt.Color("bucket:N", title="Want to Read count",
                  scale=alt.Scale(domain=load.BUCKET_ORDER, range=theme.series_colours(load.BUCKET_ORDER, None)),
                  legend=alt.Legend(orient="top"))
        if colour_by == BY_BUCKET else alt.value(theme.INK)
    ),
    tooltip=tips,
)
diagonal = alt.Chart(pd.DataFrame({"x": [0, upper], "y": [0, upper], "line": ["exact prediction"] * 2})).mark_line(
    color=theme.INK, strokeWidth=1, clip=True
).encode(x="x:Q", y="y:Q", tooltip=[alt.Tooltip("line:N", title="Diagonal")])
left, right = st.columns([2, 1], gap="large")
with left:
    theme.show_chart((points + diagonal).properties(height=520).interactive())
with right:
    st.markdown(
        f"A fixed sample of {len(sample):,} predictions from the tuned HistGB on {short.lower()}, each made by a "
        "model trained without that book's author. Points on the diagonal are exact."
        + (f" {outside} points above {upper:.0f} are not drawn." if outside else "")
    )
    st.markdown("Scroll on the chart to zoom and drag to pan; double-click to reset.")

st.header("Error by Want to Read count")
tuned_all = load.coverage_table(metrics, target, [load.SHIPPED_FAMILY])
tuned = tuned_all.set_index("bucket")["mae"]
st.markdown(
    f"MAE for books grouped by how many people shelved them as Want to Read, for the models chosen above. "
    f"For the tuned HistGB it is lowest at {tuned.idxmin()} ({tuned.min():.4f}) and highest at "
    f"{tuned.idxmax()} ({tuned.max():.4f})."
)
if chosen:
    coverage = load.coverage_table(metrics, target, chosen)
    coverage["name"] = [family(f) for f in coverage["family"]]
    chosen_names = [family(f) for f in chosen]
    left, right = st.columns(2, gap="large")
    with left:
        cov_chart = alt.Chart(coverage).mark_line(point=True).encode(
            x=alt.X("bucket:N", sort=load.BUCKET_ORDER, title="Want to Read count"),
            y=alt.Y("mae:Q", title="MAE"),
            color=alt.Color(
                "name:N",
                title="Model",
                scale=alt.Scale(domain=chosen_names,
                                range=theme.series_colours(chosen_names, family(shipped_name) if shipped_name else None)),
                legend=alt.Legend(orient="top", columns=2),
            ),
            tooltip=[alt.Tooltip("name:N", title="Model"), alt.Tooltip("bucket:N", title="Want to Read count"),
                     alt.Tooltip("mae:Q", title="MAE", format=".4f")],
        ).properties(height=320)
        theme.show_chart(cov_chart)
    with right:
        wide = load.coverage_wide(coverage.drop(columns="name")).rename(columns={f: family(f) for f in chosen})
        ui.static_table(
            wide, labels={"bucket": "Want to Read count"}, formats={n: "{:.4f}" for n in chosen_names},
            numeric=chosen_names,
        )

st.header("Sanity checks")
sanity = cached.json_file("sanity_checks_day6.json")
st.markdown(
    "Four expectations were written down before the numbers were run, then checked against the data "
    "and against the app's model's predictions for books by unseen authors."
)
expectations = load.expectations_table(sanity)
for column in ("held_truth", "held_pred"):
    expectations[column] = expectations[column].map({True: "held", False: "did not hold"})
left, right = st.columns(2, gap="large")
with left:
    ui.static_table(
        expectations,
        labels={"expectation": "Expectation", "rule": "Rule", "held_truth": "Held in the data",
                "held_pred": "Held in the predictions"},
    )
with right:
    bands = load.group_means_table(sanity["page_bands"])
    ui.static_table(
        bands,
        labels={"tag": "Page band", "n_works": "Books", "mean_truth": "Mean share, actual",
                "mean_pred": "Mean share, predicted", "mae": "MAE"},
        formats={"n_works": "{:,}", "mean_truth": "{:.4f}", "mean_pred": "{:.4f}", "mae": "{:.4f}"},
        numeric=["n_works", "mean_truth", "mean_pred", "mae"],
    )
    gap = sanity["max_gap"]
    st.markdown(
        f"Across every group checked, page bands and subject groups, the largest gap between the actual and "
        f"predicted mean share is {gap['gap']:.4f}, for {gap['row']}."
    )

st.header("Background (ratio target, default HistGB)")
st.markdown(
    "The next three sections study the model that led after day 4 of the project: HistGB with default "
    "settings, trained on ratio. They were built to explain that model's limits and were not rerun on the "
    "app's model, so the target toggle does not change them."
)

st.subheader("MAE history on ratio")
st.markdown(
    "How the error on the older ratio target fell over the project, from the first baseline (day 3) to the "
    "full model search (day 5)."
)
history = load.history_table(
    cached.json_file("metrics_day3.json"), cached.json_file("metrics_day4.json"), metrics
)
history["stage"] = history["stage"].replace(HISTORY_LABELS)
left, right = st.columns(2, gap="large")
with left:
    hist_chart = alt.Chart(history).mark_bar(color=theme.NEUTRAL_FILL).encode(
        x=alt.X("mae:Q", title="MAE on ratio"),
        y=alt.Y("stage:N", sort=list(history["stage"]), title=None),
        tooltip=[alt.Tooltip("stage:N", title="Stage"), alt.Tooltip("mae:Q", title="MAE", format=".4f")],
    ).properties(height=48 * len(history))
    theme.show_chart(hist_chart)
with right:
    ui.static_table(
        history,
        labels={"stage": "Stage", "mae": "MAE"},
        formats={"mae": "{:.4f}"},
        numeric=["mae"],
    )

st.subheader("Learning curve")
curve_json = cached.json_file("learning_curve.json")
curve = load.learning_curve_table(curve_json)
st.markdown(
    f"The day 4 model, HistGB with default settings on ratio, trained on growing shares of the authors. "
    f"R squared moves from {curve['r2'].iloc[0]:.4f} at {curve['fraction'].iloc[0]:.0%} to "
    f"{curve['r2'].iloc[-1]:.4f} at {curve['fraction'].iloc[-1]:.0%}: more books of the same kind add little."
)


def curve_chart(column: str, title: str) -> alt.Chart:
    """Line of one learning-curve measure against training books.

    :param column: "mae" or "r2".
    :param title: y-axis title.
    :return: the chart.
    """
    return alt.Chart(curve).mark_line(point=True, color=theme.NEUTRAL_FILL).encode(
        x=alt.X("mean_train_rows:Q", title="Books in training (average per fold)"),
        y=alt.Y(f"{column}:Q", title=title, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("fraction:Q", title="Share of authors", format=".0%"),
                 alt.Tooltip("mean_train_rows:Q", title="Books in training", format=","),
                 alt.Tooltip(f"{column}:Q", title=title, format=".4f")],
    ).properties(height=280)


left, right = st.columns(2, gap="large")
with left:
    theme.show_chart(curve_chart("mae", "MAE"))
with right:
    theme.show_chart(curve_chart("r2", "R squared"))
notes.add(
    "Learning curve",
    "Method: HistGB with default settings, fitted on the log of 1 plus ratio, tested five times by first "
    "author. At each step only a share of the training authors is kept (whole authors, fixed seed); the test "
    "fold stays complete. MAE and R squared are pooled over the five test folds.",
)

st.subheader("Why R squared stops where it does")
r2 = cached.json_file("r2_investigation.json")
tail = r2["variance_breakdown"]["top_1pct_largest_ratio"]
trimmed = r2["variance_breakdown"]["r2_excluding_ratio_gt_2"]
n_books = r2["n_rows"]
st.markdown(
    "This check reran the day 4 model, HistGB with default settings. The unexplained variation sits in the "
    "long tail and in books with few shelvings."
)
t1, t2, t3 = st.columns([1, 1, 2], gap="large")
with t1:
    st.metric(
        "Squared error from the 1% of books with the largest ratio",
        f"{tail['share_of_total_squared_error']:.0%}",
        help=f"The {tail['n_rows']:,} books (1%) with the highest ratio.",
    )
with t2:
    st.metric(
        f"R squared without the {trimmed['n_rows_dropped']} books above ratio 2",
        f"{trimmed['r2']:.4f}",
        delta=f"from {r2['day4_reproduction']['r2_ratio_scale']:.4f} on all books",
        delta_color="off",
        delta_arrow="off",
        help=f"R squared on the {n_books - trimmed['n_rows_dropped']:,} books with ratio 2 or less, "
        "using the same predictions.",
    )
with t3:
    buckets = load.r2_bucket_table(r2)
    ui.static_table(
        buckets,
        labels={"bucket": "Want to Read count", "ratio": "R squared, ratio", "share": "R squared, share"},
        formats={"ratio": "{:.4f}", "share": "{:.4f}"},
        numeric=["ratio", "share"],
    )
ceiling = r2["noise_ceiling"]
st.markdown(
    f"An estimate of the best R squared that counting noise should allow (a binomial noise ceiling) came "
    f"out below what the default HistGB already reaches ({ceiling['share_ceiling_overall']:.4f} against "
    f"{r2['share_model']['r2_overall']:.4f} on share, {ceiling['ratio_ceiling_overall']:.4f} against "
    f"{r2['day4_reproduction']['r2_ratio_scale']:.4f} on ratio), so the estimate cannot be right and is kept "
    "only as a negative result."
)

st.header("Personal layer (synthetic readers)")
caveat = content.CAVEATS["synthetic_readers"]
with st.container(border=True):
    st.markdown(f"**Synthetic data.** {caveat}")

blend = cached.json_file("blend_weight_day7.json")
st.markdown(
    f"The app blends the population share with the reader's own finish rate. A reader with "
    f"{blend['min_books_for_weight']} or more finished or abandoned books gets a weight fitted by "
    f"leave-one-out; others get the default {blend['default_weight']}. The weight is how much the blend trusts "
    "the population figure: 1 means only the population, 0 only the reader. Before fitting, the population "
    "figure is moved to the reader's own finish level; the level adjustment column is that move. Errors are "
    "the Brier score: the mean squared error against finished (1) or not (0)."
)
personas = load.personas_table(blend)
personas["persona"] = personas["persona"].replace(names.PERSONAS)
ui.static_table(
    personas,
    labels={"persona": "Persona", "weight": "Learned weight", "weight_raw": "Weight without the level adjustment",
            "n_books": "Finished or abandoned books", "pop_shift": "Level adjustment (logit)",
            "error_at_weight": "Brier, learned", "error_at_default": "Brier, default",
            "error_population_only": "Brier, population only"},
    formats={"weight": "{:.1f}", "weight_raw": "{:.1f}", "n_books": "{:,}", "pop_shift": "{:.2f}",
             "error_at_weight": "{:.4f}", "error_at_default": "{:.4f}", "error_population_only": "{:.4f}"},
    numeric=["weight", "weight_raw", "n_books", "pop_shift", "error_at_weight", "error_at_default",
             "error_population_only"],
)

profile = cached.json_file("profile_features_day7.json")
study = content.CAVEATS["profile_study"]
st.subheader("Profile feature study")
ui.explain("log_loss")
with st.container(border=True):
    st.markdown(f"**Synthetic data.** {study}")
st.markdown(
    f"{profile['n_readers']} simulated readers, {profile['n_books']} books each, {profile['n_rows']:,} reader-book "
    f"pairs. The population share alone scores log loss {profile['base']['log_loss']:.4f} and Brier "
    f"{profile['base']['brier']:.4f}; each row below adds one fact about the reader to it. Lower is better for "
    "both, so a gain is how much each one falls."
)
marginal = load.profile_gain_table(profile, "marginal")
marginal["name"] = marginal["feature"].replace(names.PROFILE_FEATURES)
GAINS = ["log_loss", "log_loss_gain", "brier", "brier_gain"]
ui.static_table(
    marginal,
    order=["name", *GAINS],
    labels={"name": "Reader fact added", "log_loss": "Log loss", "log_loss_gain": "Log loss gain",
            "brier": "Brier", "brier_gain": "Brier gain"},
    formats={c: gain_text for c in GAINS},
    numeric=GAINS,
)

notes.render()
