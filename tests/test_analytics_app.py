"""Smoke test: every analytics dashboard page runs on the real data without an exception."""

import os
from pathlib import Path

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1",
    reason="streamlit.testing.v1.AppTest is not available in this Streamlit version",
).AppTest

ANALYTICS_DIR = Path(__file__).resolve().parent.parent / "analytics"
APP_PATH = str(ANALYTICS_DIR / "app.py")
PROCESSED = Path(os.getenv("CAPSTONE_DATA") or ANALYTICS_DIR / "data") / "processed"
NEEDED = ["metrics_day5.json", "training.parquet", "llm_eval.json", "sweep_day5_share_oof.parquet"]
PAGES = [
    "pages/overview.py",
    "pages/data.py",
    "pages/models.py",
    "pages/features.py",
    "pages/llm_evaluation.py",
    "pages/stack.py",
]

pytestmark = pytest.mark.skipif(
    not all((PROCESSED / name).exists() for name in NEEDED),
    reason=f"analytics data not found under {PROCESSED}",
)


@pytest.fixture
def app(monkeypatch) -> AppTest:
    """The dashboard with its own folder importable, as `streamlit run app.py` makes it."""
    monkeypatch.syspath_prepend(str(ANALYTICS_DIR))
    monkeypatch.chdir(ANALYTICS_DIR)
    return AppTest.from_file(APP_PATH, default_timeout=120)


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(app, page):
    app.run()
    assert not app.exception, app.exception
    app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.title, f"{page} shows no title"


def test_models_page_ratio_toggle(app):
    app.run()
    app.switch_page("pages/models.py").run()
    app.segmented_control(key="models_target").set_value("ratio").run()
    assert not app.exception, [e.value for e in app.exception]


def _open(app, page):
    app.run()
    app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _ok(app):
    assert not app.exception, [e.value for e in app.exception]


def test_data_page_controls(app):
    _open(app, "pages/data.py")
    app.selectbox(key="data_target").set_value("ratio").run()
    _ok(app)
    app.slider(key="data_bins").set_value(80).run()
    _ok(app)
    app.multiselect(key="data_features").set_value(["log1p_n_want", "pages_median"]).run()
    _ok(app)
    assert "Stopped Reading" not in app.selectbox(key="data_shelf").options  # too few events
    app.selectbox(key="data_shelf").set_value("Currently Reading").run()
    _ok(app)


def test_models_page_controls(app):
    _open(app, "pages/models.py")
    app.selectbox(key="models_measure").set_value("r2").run()
    _ok(app)
    app.multiselect(key="models_families").set_value(["ridge", "hist_gb"]).run()
    _ok(app)
    app.multiselect(key="models_families").set_value([]).run()
    _ok(app)
    app.slider(key="models_points").set_value(20000).run()
    _ok(app)
    app.selectbox(key="models_colour").set_value("Want to Read count").run()
    _ok(app)


def test_features_page_controls(app):
    _open(app, "pages/features.py")
    app.selectbox(key="features_method").set_value("shap_mean_abs").run()
    _ok(app)
    app.slider(key="features_top").set_value(35).run()
    _ok(app)
    app.selectbox(key="features_group").set_value("Editions").run()
    _ok(app)


def test_llm_page_controls(app):
    _open(app, "pages/llm_evaluation.py")
    app.selectbox(key="llm_metric").set_value("mean_cost_usd").run()
    _ok(app)
    app.selectbox(key="llm_round").set_value("r4").run()
    _ok(app)
