"""Tests for analytics/load.py, the dashboard's table builders, on small fixtures."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analytics"))

import load  # noqa: E402


def _family(mae: float, rmse: float | None, coverage: dict | None = None, **extra) -> dict:
    """One metrics_day5.json family entry with the keys the builders read."""
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": extra.get("r2", 0.5),
        "spearman": extra.get("spearman", 0.8),
        "mae_std": extra.get("mae_std", 0.001),
        "coverage": coverage or {"50+": 0.04, "5-9": 0.06, "20-49": 0.05, "10-19": 0.07},
        "arm": extra.get("arm", {"counts": "log1p", "target": "logit"}),
    }


@pytest.fixture
def metrics() -> dict:
    """A two-target metrics_day5.json with four families each, out of MAE order."""
    families = {
        "median": _family(0.12, 0.14, spearman=None),
        "hist_gb": _family(0.06, 0.08),
        "catboost": _family(0.059, 0.079),
        "hist_gb_default": _family(0.0625, 0.082),
    }
    ratio = {k: _family(v["mae"] * 2, v["rmse"] * 2, arm={"counts": "raw", "target": "log1p"}) for k, v in families.items()}
    return {
        "share": {"n_rows": 100, "best_single": "catboost", "families": families},
        "ratio": {"n_rows": 100, "best_single": "catboost", "families": ratio},
    }


def test_mse_is_rmse_squared():
    assert load.mse_from_rmse(0.08113092069098811) == pytest.approx(0.08113092069098811**2, rel=0, abs=1e-15)
    assert load.mse_from_rmse(2.0) == 4.0


def test_mse_of_missing_rmse_is_nan():
    assert math.isnan(load.mse_from_rmse(None))


def test_mse_from_rmse_matches_pooled_mse():
    y = np.array([0.1, 0.4, 0.2, 0.9])
    pred = np.array([0.2, 0.3, 0.2, 0.5])
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    assert load.mse_from_rmse(rmse) == pytest.approx(float(np.mean((y - pred) ** 2)))


def test_improvement_is_fractional_reduction():
    assert load.improvement(0.06, 0.12) == pytest.approx(0.5)


def test_families_table_sorted_by_mae_with_mse(metrics):
    table = load.families_table(metrics, "share")
    assert list(table["family"]) == ["catboost", "hist_gb", "hist_gb_default", "median"]
    assert table["mse"].tolist() == pytest.approx((table["rmse"] ** 2).tolist())
    assert math.isnan(table.loc[table["family"] == "median", "spearman"].iloc[0])


def test_families_table_marks_shipped_only_on_share(metrics):
    share = load.families_table(metrics, "share")
    ratio = load.families_table(metrics, "ratio")
    assert share.loc[share["shipped"], "family"].tolist() == ["hist_gb"]
    assert not ratio["shipped"].any()


def test_family_notes():
    assert load.family_note("share", "hist_gb", "catboost") == "the app's model"
    assert load.family_note("share", "median", "catboost") == "predicts the median for every book"
    assert load.family_note("share", "hist_gb_default", "catboost") == "default settings, for comparison"
    assert load.family_note("share", "catboost", "catboost") == "most accurate single model, not used"
    assert load.family_note("ratio", "hist_gb", "catboost") == "the app's model (share only)"
    assert load.family_note("share", "ridge", "catboost") == ""


def test_order_buckets_is_numeric_not_lexical():
    assert load.order_buckets(["50+", "10-19", "5-9", "20-49"]) == ["5-9", "10-19", "20-49", "50+"]
    assert sorted(["50+", "10-19", "5-9", "20-49"]) != load.BUCKET_ORDER  # plain sort gets it wrong


def test_bucket_sort_key_rejects_non_bucket():
    with pytest.raises(ValueError):
        load.bucket_sort_key("n/a")


def test_coverage_table_orders_buckets_within_family(metrics):
    table = load.coverage_table(metrics, "share", ["hist_gb", "median"])
    assert table["bucket"].tolist() == load.BUCKET_ORDER * 2
    assert table["family"].tolist() == ["hist_gb"] * 4 + ["median"] * 4


def test_coverage_wide_keeps_bucket_and_family_order(metrics):
    wide = load.coverage_wide(load.coverage_table(metrics, "share", ["median", "hist_gb"]))
    assert wide["bucket"].tolist() == load.BUCKET_ORDER
    assert list(wide.columns) == ["bucket", "median", "hist_gb"]
    assert wide.loc[0, "hist_gb"] == 0.06


def test_history_table_reads_each_stage(metrics):
    day3 = {"median_baseline": {"mean_mae": 0.2386}, "histgb": {"mean_mae": 0.1494}}
    day4 = {"families": {"hist_gb": {"mean_mae": 0.1440}}}
    table = load.history_table(day3, day4, metrics)
    assert table["stage"].tolist()[0] == "Day 3 median baseline"
    assert table["mae"].tolist() == pytest.approx([0.2386, 0.1494, 0.1440, 0.12, 0.118])


def test_learning_curve_sorted_by_fraction():
    curve = {"fractions": {"1.0": {"mean_train_rows": 100, "mae": 0.1, "r2": 0.5, "seconds": 9.0},
                           "0.1": {"mean_train_rows": 10, "mae": 0.2, "r2": 0.4, "seconds": 1.0}}}
    table = load.learning_curve_table(curve)
    assert table["fraction"].tolist() == [0.1, 1.0]
    assert table["mean_train_rows"].tolist() == [10, 100]


def test_oof_sample_is_fixed_and_capped():
    oof = pd.DataFrame({"y": np.arange(50) / 50, "hist_gb": np.arange(50) / 49})
    a = load.oof_sample(oof, "hist_gb", n=10, seed=1)
    b = load.oof_sample(oof, "hist_gb", n=10, seed=1)
    assert list(a.columns) == ["actual", "predicted"] and len(a) == 10
    pd.testing.assert_frame_equal(a, b)
    assert len(load.oof_sample(oof, "hist_gb", n=500)) == 50


def test_histogram_counts_and_overflow():
    values = pd.Series([0.1, 0.2, 0.6, 0.9, 1.5, None])
    hist = load.histogram_table(values, bins=2, upper=1.0)
    assert hist["count"].tolist() == [2, 2]
    assert hist["start"].tolist() == [0.0, 0.5]
    assert load.count_above(values, 1.0) == 1


def test_missing_table_drops_complete_columns_and_sorts():
    df = pd.DataFrame({"a": [1, None, None, 4], "b": [1, 2, 3, 4], "c": [None, 2, 3, 4]})
    table = load.missing_table(df)
    assert table["column"].tolist() == ["a", "c"]
    assert table["n_missing"].tolist() == [2, 1]
    assert table["share_missing"].tolist() == [0.5, 0.25]


def test_spearman_is_rank_based():
    x = pd.Series([1, 2, 3, 4, 5])
    assert load.spearman(x, x**3) == pytest.approx(1.0)
    assert load.spearman(x, -x) == pytest.approx(-1.0)
    assert math.isnan(load.spearman(x, pd.Series([2, 2, 2, 2, 2])))


def test_spearman_table_ranks_by_absolute_first_target():
    df = pd.DataFrame({"share": [1, 2, 3, 4, 5], "ratio": [2, 1, 4, 3, 5],
                       "up": [1, 2, 3, 5, 4], "down": [5, 4, 3, 2, 1], "flat": [1, 3, 2, 3, 1]})
    table = load.spearman_table(df, ["flat", "up", "down"], ["share", "ratio"], top=2)
    assert table["feature"].tolist() == ["down", "up"]
    assert list(table.columns) == ["feature", "share", "ratio"]


def test_seasonal_profile_averages_normalised_years():
    rows = []
    for year, counts in [(2018, [1, 3]), (2019, [10, 10])]:
        for month, c in zip([1, 2], counts):
            rows.append({"year": year, "month": month, "shelf": "Already Read", "count": c})
    rows.append({"year": 2020, "month": 1, "shelf": "Already Read", "count": 999})  # outside years
    profile = load.seasonal_profile(pd.DataFrame(rows), ["Already Read"], range(2018, 2020))
    assert profile.sort_values("month")["share"].tolist() == pytest.approx([(0.25 + 0.5) / 2, (0.75 + 0.5) / 2])


def test_complete_years_needs_january_and_december():
    rows = [{"year": y, "month": m, "shelf": "s", "count": 1} for y in (2017, 2018, 2019) for m in (1, 12)]
    rows = [r for r in rows if not (r["year"] == 2017 and r["month"] == 1)]
    assert load.complete_years(pd.DataFrame(rows)) == range(2018, 2020)


SVD = {"svd_10": "ten", "svd_2": "two", "svd_0": "zero"}


def test_model_features_swaps_subject_indicators_for_svd():
    stored = ["log1p_n_want", "subj_fiction", "pages_median", "subj_history"]
    assert load.model_features(stored, SVD) == ["log1p_n_want", "pages_median", "svd_0", "svd_2", "svd_10"]


def test_feature_groups_table_and_ungrouped():
    features = ["log1p_n_want", "pages_median", "svd_0", "svd_2"]
    groups = [{"group": "Counts", "columns": ["log1p_n_want"]}, {"group": "Subjects", "columns": "svd"}]
    table = load.feature_groups_table(features, groups, SVD)
    assert table["feature"].tolist() == ["log1p_n_want", "svd_0", "svd_2"]
    assert table["terms"].tolist() == ["", "zero", "two"]
    assert load.ungrouped(features, table) == ["pages_median"]


def test_importance_table_labels_svd_and_sorts():
    entries = [{"name": "svd_0", "mean": 0.1, "std": 0.0}, {"name": "log1p_n_want", "mean": 0.5, "std": 0.1},
               {"name": "n_covers", "mean": 0.01, "std": 0.0}]
    table = load.importance_table(entries, SVD, top=2)
    assert table["feature"].tolist() == ["log1p_n_want", "svd_0"]
    assert table["label"].tolist() == ["log1p_n_want", "svd_0 (zero)"]


def test_arm_table_side_by_side(metrics):
    table = load.arm_table(metrics, "hist_gb", ["share", "ratio"])
    assert table.to_dict("records") == [
        {"setting": "counts", "share": "log1p", "ratio": "raw"},
        {"setting": "target", "share": "logit", "ratio": "log1p"},
    ]


def test_r2_bucket_table_orders_buckets():
    r2 = {"variance_breakdown": {"r2_by_n_want_bucket": {"50+": 0.5, "5-9": 0.2, "10-19": 0.3}},
          "share_model": {"r2_by_n_want_bucket": {"5-9": 0.3, "50+": 0.6, "10-19": 0.4}}}
    table = load.r2_bucket_table(r2)
    assert table["bucket"].tolist() == ["5-9", "10-19", "50+"]
    assert table["share"].tolist() == [0.3, 0.4, 0.6]


def test_expectations_and_personas_tables():
    sanity = {"expectations": [{"name": "(a)", "rule": "r", "held_truth": True, "held_pred": False, "x": 1}]}
    assert load.expectations_table(sanity).columns.tolist() == ["expectation", "rule", "held_truth", "held_pred"]
    keys = ["weight", "weight_raw", "n_books", "pop_shift", "error_at_weight", "error_at_default", "error_population_only"]
    blend = {"personas": {"p1": {k: 1 for k in keys} | {"curve": {}}}}
    table = load.personas_table(blend)
    assert table.columns.tolist() == ["persona", *keys] and table["persona"].tolist() == ["p1"]


def _fig(value):
    return {"value": value, "route": "computed", "source": "s", "md_text": None, "md_line": None, "md_matches": None}


def test_llm_tables():
    llm = {
        "rounds": [{"round": "r1", "variant": "A", "prompt_version": "abc (first 8)", "judge_pass": 10},
                   {"round": "r1", "variant": "B", "prompt_version": "abc (all 22)", "judge_pass": 9},
                   {"round": "r2", "variant": "A", "prompt_version": "def", "judge_pass": 12}],
        "judges": [{"round": "r4",
                    "per_judge": {"opus": {"real_pass": _fig(10)}, "sonnet": {"real_pass": _fig(12)}},
                    "agreement": {"cohens_kappa": _fig(0.673)}}],
    }
    assert load.llm_rounds_table(llm)["label"].tolist() == ["Round 1 (8 examples)", "Round 1 (all 22 examples)", "Round 2"]
    judges = load.llm_judge_table(llm)
    assert judges[["judge", "real_pass", "cohens_kappa"]].values.tolist() == [["opus", 10, 0.673], ["sonnet", 12, 0.673]]


def test_readers_use_processed_folder(tmp_path):
    folder = tmp_path / "processed"
    folder.mkdir()
    (folder / "x.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    pd.DataFrame({"y": [1.0], "z": [2.0]}).to_parquet(folder / "t.parquet")
    assert load.processed_path("x.json", tmp_path) == folder / "x.json"
    assert load.read_json(load.processed_path("x.json", tmp_path)) == {"a": 1}
    assert load.read_parquet(folder / "t.parquet", ["z"]).columns.tolist() == ["z"]


def test_rank_by_metric_puts_best_first_and_missing_last(metrics):
    table = load.families_table(metrics, "share")
    assert load.rank_by_metric(table, "mae")["family"].tolist()[0] == "catboost"
    table.loc[table["family"] == "hist_gb", "r2"] = 0.9
    ranked = load.rank_by_metric(table, "r2")
    assert ranked["family"].iloc[0] == "hist_gb"
    by_rho = load.rank_by_metric(table, "spearman")
    assert by_rho["family"].iloc[-1] == "median"  # spearman None sorts last
    assert list(by_rho.index) == list(range(len(by_rho)))


def test_filter_rows_keeps_table_order():
    df = pd.DataFrame({"family": ["a", "b", "c"], "x": [1, 2, 3]})
    assert load.filter_rows(df, "family", ["c", "a"])["family"].tolist() == ["a", "c"]
    assert load.filter_rows(df, "family", []).empty


def test_want_bucket_matches_baseline_edges():
    log1p = np.log1p(pd.Series([4, 5, 9, 10, 19, 20, 49, 50, 500]))
    assert load.want_bucket(log1p).tolist() == [None, "5-9", "5-9", "10-19", "10-19", "20-49", "20-49", "50+", "50+"]


def test_attach_bucket_joins_on_work_key_and_keeps_order():
    oof = pd.DataFrame({"work_key": ["w2", "w1", "w9"], "y": [0.1, 0.2, 0.3]})
    training = pd.DataFrame({"work_key": ["w1", "w2"], "log1p_n_want": np.log1p([7, 60])})
    out = load.attach_bucket(oof, training)
    assert out["work_key"].tolist() == ["w2", "w1", "w9"]
    assert out["bucket"].tolist()[:2] == ["50+", "5-9"] and pd.isna(out["bucket"].iloc[2])


def test_oof_sample_carries_bucket_and_same_rows_as_before():
    oof = pd.DataFrame({"y": np.arange(50) / 50, "hist_gb": np.arange(50) / 49})
    with_bucket = oof.assign(bucket=["5-9"] * 50)
    a = load.oof_sample(oof, "hist_gb", n=10, seed=3)
    b = load.oof_sample(with_bucket, "hist_gb", n=10, seed=3)
    assert "bucket" not in a and b["bucket"].tolist() == ["5-9"] * 10
    pd.testing.assert_frame_equal(a, b[["actual", "predicted"]])


def test_distribution_summary_rows():
    values = pd.Series([0.5, 1.0, 2.0, 4.0, None])
    table = load.distribution_summary(values, upper=3.0)
    assert table["measure"].tolist() == ["Works", "Median", "Mean", "Maximum", "Above 3, not drawn"]
    assert table["value"].tolist() == pytest.approx([4, 1.5, 1.875, 4.0, 1])
    assert len(load.distribution_summary(values)) == 4


def test_spearman_table_without_top_keeps_all():
    df = pd.DataFrame({"share": [1, 2, 3, 4], "ratio": [1, 2, 3, 4], "a": [4, 3, 2, 1], "b": [1, 3, 2, 4]})
    assert load.spearman_table(df, ["b", "a"], ["share", "ratio"])["feature"].tolist() == ["a", "b"]


def test_llm_verdicts_table():
    llm = {"verdicts": [{"round": "r4", "source": "s", "items": [
        {"id": "J01", "kind": "real", "work_key": "OL1W", "title": "T",
         "judges": {"opus": {"verdict": "PASS", "failed": [], "hook": "yes"},
                    "sonnet": {"verdict": "FAIL", "failed": ["rules_ok", "faithful"], "hook": "partly"}}}]}]}
    row = load.llm_verdicts_table(llm, "r4").iloc[0]
    assert row["opus_verdict"] == "PASS" and row["opus_failed"] == ""
    assert row["sonnet_failed"] == "rules_ok, faithful" and row["sonnet_hook"] == "partly"


def _ship_metrics() -> dict:
    """A share block with the shipped HistGB and three rivals, fold MAEs included."""
    def fam(mae, folds, fit, arm):
        return {"mae": mae, "fold_maes": folds, "fit_seconds": fit, "arm": arm}

    return {"share": {"families": {
        "hist_gb": fam(0.0603, [0.060, 0.061, 0.060, 0.060, 0.060], 5.2, {"year": "raw"}),
        "catboost": fam(0.0594, [0.059, 0.060, 0.059, 0.059, 0.059], 362.7, {"year": "era", "cat_subject": True}),
        "xgboost": fam(0.0596, [0.059, 0.061, 0.061, 0.059, 0.059], 12.2, {"year": "era"}),
        "lightgbm": fam(0.0605, [0.061, 0.062, 0.061, 0.061, 0.059], 1.9, {"year": "raw"}),
    }}}


def test_extra_step_names_each_change_or_none():
    assert load.extra_step({"year": "raw"}) == "none"
    assert load.extra_step({"year": "era"}) == "publish year recoded as era columns"
    assert load.extra_step({"year": "era", "cat_subject": True}) == (
        "the first subject as a category; publish year recoded as era columns"
    )


def test_ship_comparison_gaps_folds_and_shipped_row():
    table = load.ship_comparison(_ship_metrics()).set_index("family")
    assert list(table.index) == ["hist_gb", "catboost", "xgboost", "lightgbm"]
    assert table.loc["hist_gb", "shipped"] and pd.isna(table.loc["hist_gb", "gap"])
    assert table.loc["catboost", "gap"] == pytest.approx(-0.0009)
    assert table.loc["lightgbm", "gap"] == pytest.approx(0.0002)
    # Lower MAE on a fold counts; a tie or a higher fold does not.
    assert table.loc["catboost", "folds_lower"] == 5
    assert table.loc["xgboost", "folds_lower"] == 3
    assert table.loc["lightgbm", "folds_lower"] == 1
