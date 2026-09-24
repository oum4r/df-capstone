"""Load the processed data and build every table the dashboard shows.

Apart from read_json and read_parquet every function is pure, so the tests
can feed it small fixtures; Streamlit caching lives in cached.py.

DATA_DIR is CAPSTONE_DATA when set, otherwise the copy bundle_data.py writes
to analytics/data.
"""

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import names

BUNDLED_DATA = Path(__file__).resolve().parent / "data"


def resolve_data_dir(env: str | None, bundled: Path = BUNDLED_DATA) -> Path:
    """The data root: CAPSTONE_DATA when set, otherwise the bundled copy.

    :param env: the CAPSTONE_DATA value, or None when unset.
    :param bundled: the bundled data root next to this file.
    :return: the data root to read from.
    """
    return Path(env) if env else bundled


DATA_DIR = resolve_data_dir(os.getenv("CAPSTONE_DATA"))
PROCESSED = "processed"

SHIPPED_TARGET = "share"
SHIPPED_FAMILY = "hist_gb"
BASELINE_FAMILY = "median"
DEFAULT_FAMILY = "hist_gb_default"

# Want-to-read bucket labels as metrics_day5.json writes them, and the edges the model training used.
BUCKET_ORDER = ["5-9", "10-19", "20-49", "50+"]
BUCKET_EDGES = [5, 10, 20, 50, np.inf]
# Measures where a larger value is better; every other measure is an error.
HIGHER_IS_BETTER = {"r2", "spearman"}


def processed_path(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Path of one file under DATA_DIR/processed.

    :param name: file name, e.g. "metrics_day5.json".
    :param data_dir: the data root; defaults to DATA_DIR.
    :return: the full path.
    """
    return Path(data_dir) / PROCESSED / name


def read_json(path: Path) -> dict:
    """Read one JSON file as UTF-8.

    :param path: file path.
    :return: the parsed object.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_parquet(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a parquet file, optionally only some columns.

    :param path: file path.
    :param columns: column names to read, or None for all.
    :return: the table.
    """
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def mse_from_rmse(rmse: float | None) -> float:
    """MSE from a stored RMSE; exact, as the stored RMSE is the root of the pooled out-of-fold MSE.

    :param rmse: root mean squared error, or None.
    :return: rmse squared, or NaN when rmse is None.
    """
    return float("nan") if rmse is None else float(rmse) ** 2


def improvement(model_mae: float, baseline_mae: float) -> float:
    """Fractional MAE reduction against a baseline (0.48 means 48% lower).

    :param model_mae: the model's MAE.
    :param baseline_mae: the baseline's MAE, above zero.
    :return: 1 - model_mae / baseline_mae.
    """
    return 1.0 - model_mae / baseline_mae


def family_note(target: str, family: str, best_single: str | None) -> str:
    """Short label for a model family row in the families table.

    :param target: "share" or "ratio".
    :param family: family key from metrics_day5.json.
    :param best_single: that target's best_single key from the same file.
    :return: a label, or "" for an unlabelled family.
    """
    if target == SHIPPED_TARGET and family == SHIPPED_FAMILY:
        return "the app's model"
    if family == BASELINE_FAMILY:
        return "predicts the median for every book"
    if family == DEFAULT_FAMILY:
        return "default settings, for comparison"
    if family == best_single:
        return "most accurate single model, not used"
    if family == SHIPPED_FAMILY:
        return "the app's model (share only)"
    return ""


def families_table(metrics: Mapping, target: str) -> pd.DataFrame:
    """One row per model family for one target, sorted by MAE.

    :param metrics: metrics_day5.json as a dict.
    :param target: "share" or "ratio".
    :return: columns family, note, mae, mse, rmse, r2, spearman, mae_std,
        shipped (bool).
    """
    block = metrics[target]
    best = block.get("best_single")
    rows = []
    for family, m in block["families"].items():
        rows.append(
            {
                "family": family,
                "note": family_note(target, family, best),
                "mae": m["mae"],
                "mse": mse_from_rmse(m.get("rmse")),
                "rmse": m.get("rmse"),
                "r2": m.get("r2"),
                "spearman": m.get("spearman"),
                "mae_std": m.get("mae_std"),
                "shipped": target == SHIPPED_TARGET and family == SHIPPED_FAMILY,
            }
        )
    out = pd.DataFrame(rows).astype({"spearman": float})
    return out.sort_values("mae", kind="stable").reset_index(drop=True)


# MAE margin an ensemble had to beat the best single model by; later required of every accuracy change.
SHIP_MARGIN = 0.0025
SHIP_RIVALS = ("catboost", "xgboost", "lightgbm")


def extra_step(arm: Mapping) -> str:
    """Plain words for the input changes a family's best arm makes beyond the shipped arm.

    :param arm: a family's arm dict from metrics_day5.json.
    :return: the steps, joined with "; ", or "none".
    """
    steps = []
    if arm.get("cat_subject"):
        steps.append("the first subject as a category")
    if arm.get("year") == "era":
        steps.append("publish year recoded as era columns")
    return "; ".join(steps) or "none"


def ship_comparison(
    metrics: Mapping, target: str = SHIPPED_TARGET, rivals: Sequence[str] = SHIP_RIVALS
) -> pd.DataFrame:
    """The shipped HistGB against its closest rivals on one target.

    :param metrics: metrics_day5.json as a dict.
    :param target: "share" or "ratio".
    :param rivals: families to set against the shipped one.
    :return: columns family, mae, gap (rival MAE minus shipped; negative is
        better; None on the shipped row), folds_lower (folds where the rival's
        MAE is lower; None on the shipped row), fit_seconds, extra_step, shipped.
    """
    families = metrics[target]["families"]
    base = families[SHIPPED_FAMILY]
    rows = []
    for family in (SHIPPED_FAMILY, *rivals):
        m = families[family]
        is_base = family == SHIPPED_FAMILY
        rows.append(
            {
                "family": family,
                "mae": m["mae"],
                "gap": None if is_base else m["mae"] - base["mae"],
                "folds_lower": None if is_base else sum(o < b for o, b in zip(m["fold_maes"], base["fold_maes"])),
                "fit_seconds": m.get("fit_seconds"),
                "extra_step": extra_step(m.get("arm") or {}),
                "shipped": is_base,
            }
        )
    return pd.DataFrame(rows)


def rank_by_metric(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows best first on one measure, missing values last.

    :param table: families_table output (or any table with the column).
    :param metric: e.g. "mae" (lower is better) or "r2" (higher is better).
    :return: the rows re-ordered, index reset.
    """
    ascending = metric not in HIGHER_IS_BETTER
    return table.sort_values(metric, ascending=ascending, na_position="last", kind="stable").reset_index(drop=True)


def filter_rows(df: pd.DataFrame, column: str, keep: Iterable) -> pd.DataFrame:
    """Rows whose column value is in keep, in the table's own order.

    :param df: any table.
    :param column: the column to match on.
    :param keep: values to keep.
    :return: the matching rows, index reset.
    """
    return df[df[column].isin(set(keep))].reset_index(drop=True)


def want_bucket(log1p_n_want: pd.Series) -> pd.Series:
    """Want-to-read bucket per row, counted back from the log1p feature.

    :param log1p_n_want: the stored log1p_n_want feature.
    :return: labels "5-9", "10-19", "20-49", "50+"; None below 5.
    """
    n_want = np.expm1(log1p_n_want).round()
    labels = pd.cut(n_want, bins=BUCKET_EDGES, right=False, labels=BUCKET_ORDER)
    return labels.astype(object).where(labels.notna(), None)


def attach_bucket(oof: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """Add each out-of-fold row's want-to-read bucket, joined on work_key.

    :param oof: a sweep_day5_*_oof.parquet table with work_key.
    :param training: training.parquet with work_key and log1p_n_want.
    :return: oof with a bucket column, same rows in the same order.
    """
    buckets = pd.Series(want_bucket(training["log1p_n_want"]).to_numpy(), index=training["work_key"])
    out = oof.copy()
    out["bucket"] = out["work_key"].map(buckets)
    return out


def bucket_sort_key(label: str) -> int:
    """Sort key for a want-to-read bucket label: its lower bound.

    :param label: e.g. "5-9" or "50+".
    :return: the leading integer of the label.
    """
    match = re.match(r"\d+", label)
    if match is None:
        raise ValueError(f"not a bucket label: {label!r}")
    return int(match.group())


def order_buckets(labels: Iterable[str]) -> list[str]:
    """Bucket labels in numeric order ("5-9" before "10-19", not after).

    :param labels: bucket labels in any order.
    :return: the labels sorted by lower bound.
    """
    return sorted(labels, key=bucket_sort_key)


def coverage_table(metrics: Mapping, target: str, families: Sequence[str]) -> pd.DataFrame:
    """MAE by want-to-read bucket for the chosen families, long form.

    :param metrics: metrics_day5.json as a dict.
    :param target: "share" or "ratio".
    :param families: family keys to include, in display order.
    :return: columns family, bucket, mae; buckets in numeric order.
    """
    rows = []
    for family in families:
        coverage = metrics[target]["families"][family]["coverage"]
        for bucket in order_buckets(coverage):
            rows.append({"family": family, "bucket": bucket, "mae": coverage[bucket]})
    return pd.DataFrame(rows, columns=["family", "bucket", "mae"])


def coverage_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot a coverage_table to one row per bucket, one column per family.

    :param long: coverage_table output.
    :return: columns bucket plus one per family, buckets in numeric order.
    """
    wide = long.pivot(index="bucket", columns="family", values="mae")
    wide = wide.loc[order_buckets(wide.index), list(dict.fromkeys(long["family"]))]
    return wide.reset_index().rename_axis(columns=None)


def history_table(day3: Mapping, day4: Mapping, day5: Mapping) -> pd.DataFrame:
    """MAE history on the ratio target, stage by stage.

    :param day3: metrics_day3.json.
    :param day4: metrics_day4.json.
    :param day5: metrics_day5.json.
    :return: columns stage, mae, in project order.
    """
    ratio = day5["ratio"]["families"]
    best = day5["ratio"].get("best_single", "catboost")
    rows = [
        ("Day 3 median baseline", day3["median_baseline"]["mean_mae"]),
        ("Day 3 HistGB", day3["histgb"]["mean_mae"]),
        ("Day 4 default HistGB", day4["families"]["hist_gb"]["mean_mae"]),
        ("Day 5 tuned HistGB", ratio["hist_gb"]["mae"]),
        ("CatBoost", ratio[best]["mae"]),
    ]
    return pd.DataFrame(rows, columns=["stage", "mae"])


def learning_curve_table(curve: Mapping) -> pd.DataFrame:
    """Learning curve rows, smallest training fraction first.

    :param curve: learning_curve.json.
    :return: columns fraction, mean_train_rows, mae, r2, seconds.
    """
    rows = [{"fraction": float(k), **v} for k, v in curve["fractions"].items()]
    out = pd.DataFrame(rows, columns=["fraction", "mean_train_rows", "mae", "r2", "seconds"])
    return out.sort_values("fraction").reset_index(drop=True)


def oof_sample(oof: pd.DataFrame, model: str, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """A fixed-seed sample of out-of-fold predictions against the truth.

    :param oof: a sweep_day5_*_oof.parquet table with a y column.
    :param model: the prediction column, e.g. "hist_gb".
    :param n: rows to keep (all rows if the table is smaller).
    :param seed: random_state for the sample.
    :return: columns actual, predicted, and bucket when oof has one.
    """
    rows = oof.sample(n=min(n, len(oof)), random_state=seed)
    out = pd.DataFrame({"actual": rows["y"].to_numpy(), "predicted": rows[model].to_numpy()})
    if "bucket" in rows:
        out["bucket"] = rows["bucket"].to_numpy()
    return out


def histogram_table(values: pd.Series, bins: int, upper: float | None = None) -> pd.DataFrame:
    """Equal-width histogram counts over [0, upper] or [min, max].

    :param values: numeric values.
    :param bins: number of bins.
    :param upper: right edge; values above it are left out and counted by
        count_above.
    :return: columns start, end, count.
    """
    clean = values.dropna().to_numpy(dtype=float)
    low = 0.0 if upper is not None else float(clean.min())
    high = float(upper) if upper is not None else float(clean.max())
    counts, edges = np.histogram(clean[clean <= high], bins=bins, range=(low, high))
    return pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts})


def distribution_summary(values: pd.Series, upper: float | None = None) -> pd.DataFrame:
    """Median, mean, maximum and row count of a column, plus how many sit above a cut-off.

    :param values: numeric values.
    :param upper: the histogram's right edge, or None for no cut-off row.
    :return: columns measure, value.
    """
    clean = values.dropna()
    rows = [("Works", float(len(clean))), ("Median", float(clean.median())), ("Mean", float(clean.mean())),
            ("Maximum", float(clean.max()))]
    if upper is not None:
        rows.append((f"Above {upper:g}, not drawn", float(count_above(clean, upper))))
    return pd.DataFrame(rows, columns=["measure", "value"])


def count_above(values: pd.Series, upper: float) -> int:
    """How many values sit above a histogram's right edge.

    :param values: numeric values.
    :param upper: the right edge.
    :return: count of values strictly above upper.
    """
    return int((values.dropna() > upper).sum())


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column, most missing first; complete columns dropped.

    :param df: any table.
    :return: columns column, n_missing, share_missing.
    """
    n_missing = df.isna().sum()
    n_missing = n_missing[n_missing > 0].sort_values(ascending=False, kind="stable")
    return pd.DataFrame(
        {
            "column": n_missing.index,
            "n_missing": n_missing.to_numpy(dtype=int),
            "share_missing": n_missing.to_numpy(dtype=float) / len(df),
        }
    )


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation: Pearson correlation of average ranks.

    Rows where either side is missing are dropped first.

    :param x: one numeric series.
    :param y: another, same index.
    :return: the coefficient, or NaN if either side is constant.
    """
    both = pd.concat([x, y], axis=1).dropna()
    rx = both.iloc[:, 0].rank(method="average")
    ry = both.iloc[:, 1].rank(method="average")
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_table(
    df: pd.DataFrame, features: Sequence[str], targets: Sequence[str], top: int | None = None
) -> pd.DataFrame:
    """Spearman of each feature with each target, strongest first.

    Ranked by the absolute value against the first target.

    :param df: table holding the features and targets.
    :param features: feature columns.
    :param targets: target columns, e.g. ["share", "ratio"].
    :param top: rows to keep, or None for all.
    :return: columns feature plus one per target.
    """
    rows = [{"feature": f, **{t: spearman(df[f], df[t]) for t in targets}} for f in features]
    out = pd.DataFrame(rows)
    order = out[targets[0]].abs().sort_values(ascending=False, kind="stable").index
    out = out.loc[order]
    return (out if top is None else out.head(top)).reset_index(drop=True)


def seasonal_profile(df: pd.DataFrame, shelves: Sequence[str], years: range) -> pd.DataFrame:
    """Average each complete year's share of events by month.

    Each year's monthly counts are divided by that year's total before
    averaging, so growth in the dump does not favour recent months.

    :param df: shelf_by_month.parquet, columns year, month, shelf, count.
    :param shelves: shelf names to keep.
    :param years: complete calendar years to average over.
    :return: columns month, shelf, share.
    """
    subset = df[df["year"].isin(set(years)) & df["shelf"].isin(set(shelves))].copy()
    subset["year_share"] = subset["count"] / subset.groupby(["year", "shelf"])["count"].transform("sum")
    profile = subset.groupby(["month", "shelf"])["year_share"].mean().reset_index()
    return profile.rename(columns={"year_share": "share"})


def complete_years(df: pd.DataFrame, last_month: int = 12) -> range:
    """Calendar years with events in their first month and in last_month.

    :param df: shelf_by_month.parquet table.
    :param last_month: the month a complete year must reach.
    :return: range of complete years, first to last.
    """
    active = df[df["count"] > 0]
    starts = set(active.loc[active["month"] == 1, "year"])
    ends = set(active.loc[active["month"] == last_month, "year"])
    full = sorted(starts & ends)
    return range(full[0], full[-1] + 1) if full else range(0)


def model_features(feature_columns: Sequence[str], svd_labels: Mapping[str, str]) -> list[str]:
    """The app model's inputs: the stored base columns, then the svd_ components that replace the subj_ flags.

    :param feature_columns: features.json feature_columns.
    :param svd_labels: explain_day5.json svd_labels.
    :return: base columns in stored order, then svd_ columns.
    """
    base = [c for c in feature_columns if not c.startswith("subj_")]
    return base + sorted(svd_labels, key=lambda k: int(k.split("_")[1]))


def feature_label(name: str, svd_labels: Mapping[str, str]) -> str:
    """Display label: the name, with its top subject terms for an SVD column.

    :param name: a feature name.
    :param svd_labels: explain_day5.json svd_labels.
    :return: e.g. "svd_0 (fiction / romance / general / historical)".
    """
    return f"{name} ({svd_labels[name]})" if name in svd_labels else name


def feature_groups_table(
    features: Sequence[str], groups: Sequence[Mapping], svd_labels: Mapping[str, str]
) -> pd.DataFrame:
    """Group each model feature, in the grouping's order.

    :param features: model_features output.
    :param groups: content.FEATURE_GROUPS, each with group and columns;
        a group whose columns is "svd" takes every svd_ feature.
    :param svd_labels: explain_day5.json svd_labels.
    :return: columns group, feature, terms (top SVD terms, else "").
    """
    rows = []
    for g in groups:
        members = [f for f in features if f.startswith("svd_")] if g["columns"] == "svd" else g["columns"]
        rows += [
            {"group": g["group"], "feature": f, "terms": svd_labels.get(f, "")} for f in members if f in features
        ]
    return pd.DataFrame(rows, columns=["group", "feature", "terms"])


def ungrouped(features: Sequence[str], table: pd.DataFrame) -> list[str]:
    """Model features the grouping missed (should be empty).

    :param features: model_features output.
    :param table: feature_groups_table output.
    :return: features absent from the table.
    """
    return [f for f in features if f not in set(table["feature"])]


def arm_table(metrics: Mapping, family: str, targets: Sequence[str]) -> pd.DataFrame:
    """A family's feature arm side by side for each target.

    :param metrics: metrics_day5.json.
    :param family: family key, e.g. "hist_gb".
    :param targets: e.g. ["share", "ratio"].
    :return: columns setting plus one per target, values as text.
    """
    arms = {t: metrics[t]["families"][family]["arm"] for t in targets}
    settings = list(dict.fromkeys(k for arm in arms.values() for k in arm))
    return pd.DataFrame([{"setting": s, **{t: str(arms[t].get(s, "")) for t in targets}} for s in settings])


def importance_table(entries: Sequence[Mapping], svd_labels: Mapping[str, str], top: int) -> pd.DataFrame:
    """Top entries of a permutation or SHAP list, largest mean first.

    :param entries: explain_day5.json permutation or shap_mean_abs.
    :param svd_labels: explain_day5.json svd_labels.
    :param top: rows to keep.
    :return: columns feature, label, mean, std.
    """
    out = pd.DataFrame(entries, columns=["name", "mean", "std"]).rename(columns={"name": "feature"})
    out = out.sort_values("mean", ascending=False, kind="stable").head(top).reset_index(drop=True)
    out.insert(1, "label", [feature_label(f, svd_labels) for f in out["feature"]])
    return out


def r2_bucket_table(r2: Mapping) -> pd.DataFrame:
    """R squared by want-to-read bucket for the default HistGB on each target.

    :param r2: r2_investigation.json.
    :return: columns bucket, ratio, share; buckets in numeric order.
    """
    ratio = r2["variance_breakdown"]["r2_by_n_want_bucket"]
    share = r2["share_model"]["r2_by_n_want_bucket"]
    return pd.DataFrame(
        [{"bucket": b, "ratio": ratio.get(b), "share": share.get(b)} for b in order_buckets(ratio)]
    )


def expectations_table(sanity: Mapping) -> pd.DataFrame:
    """The four sanity-check expectations and whether each held.

    :param sanity: sanity_checks_day6.json.
    :return: columns expectation, rule, held_truth, held_pred.
    """
    return pd.DataFrame(
        [
            {"expectation": e["name"], "rule": e["rule"], "held_truth": e["held_truth"], "held_pred": e["held_pred"]}
            for e in sanity["expectations"]
        ]
    )


def group_means_table(rows: Sequence[Mapping]) -> pd.DataFrame:
    """Mean truth against mean out-of-fold prediction for each group.

    :param rows: sanity_checks_day6.json groups or page_bands.
    :return: columns tag, n_works, mean_truth, mean_pred, mae.
    """
    return pd.DataFrame(rows, columns=["tag", "n_works", "mean_truth", "mean_pred", "mae"])


def personas_table(blend: Mapping) -> pd.DataFrame:
    """Learned blend weight per synthetic persona.

    :param blend: blend_weight_day7.json.
    :return: columns persona, weight, weight_raw, n_books, pop_shift,
        error_at_weight, error_at_default, error_population_only.
    """
    cols = ["weight", "weight_raw", "n_books", "pop_shift", "error_at_weight", "error_at_default", "error_population_only"]
    return pd.DataFrame([{"persona": name, **{c: p[c] for c in cols}} for name, p in blend["personas"].items()])


def profile_gain_table(profile: Mapping, part: str) -> pd.DataFrame:
    """Marginal or forward-selection gains from the profile feature study.

    :param profile: profile_features_day7.json.
    :param part: "marginal" or "forward".
    :return: the list as a table, step column first when present.
    """
    return pd.DataFrame(profile[part])


def llm_rounds_table(llm: Mapping) -> pd.DataFrame:
    """Per-round automatic and pass-count metrics from llm_eval.json.

    :param llm: llm_eval.json.
    :return: one row per round and variant, with a plain label column
        ("Round 1 (8 examples)", "Round 2", ...).
    """
    out = pd.DataFrame(llm["rounds"])
    per_round = out["round"].map(out["round"].value_counts())
    versions = out["prompt_version"] if "prompt_version" in out else pd.Series([""] * len(out))
    out.insert(0, "label", [names.round_label(r, v, n) for r, v, n in zip(out["round"], versions, per_round)])
    return out


def llm_judge_table(llm: Mapping) -> pd.DataFrame:
    """Judge figures for the judged rounds, one row per round and judge.

    Agreement figures belong to the pair, so both judges' rows carry them.

    :param llm: llm_eval.json.
    :return: columns round, judge, real_pass, real_n, controls_caught,
        controls_n, hook_score, pass_fail_agreement, cohens_kappa,
        content_agreement, content_kappa, n_items.
    """
    rows = []
    for r in llm["judges"]:
        pair = {k: r["agreement"][k]["value"] for k in r["agreement"]}
        for judge, figs in r["per_judge"].items():
            rows.append({"round": r["round"], "judge": judge, **{k: v["value"] for k, v in figs.items()}, **pair})
    return pd.DataFrame(rows)


def llm_verdicts_table(llm: Mapping, rnd: str) -> pd.DataFrame:
    """Per-item verdicts of both judges for one judged round.

    :param llm: llm_eval.json, with a verdicts list (see build_llm_eval).
    :param rnd: round id, e.g. "r4".
    :return: one row per batch item: id, kind, title, work_key, then per
        judge verdict, failed criteria (comma-joined, "" if none) and hook.
    """
    block = next(v for v in llm["verdicts"] if v["round"] == rnd)
    rows = []
    for item in block["items"]:
        row = {k: item[k] for k in ("id", "kind", "title", "work_key")}
        for judge, v in item["judges"].items():
            row[f"{judge}_verdict"] = v["verdict"]
            row[f"{judge}_failed"] = ", ".join(v["failed"])
            row[f"{judge}_hook"] = v["hook"]
        rows.append(row)
    return pd.DataFrame(rows)
