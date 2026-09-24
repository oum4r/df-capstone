"""Tests for analytics/names.py, the one code-name to display-name mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analytics"))

import names  # noqa: E402


def test_family_names_mark_the_app_model_on_share_only():
    assert names.family_name("hist_gb") == "HistGB (tuned), the app's model"
    assert names.family_name("hist_gb", "ratio") == "HistGB (tuned)"
    assert names.family_name("median") == "Median guess (baseline)"
    assert names.family_name("unknown_family") == "unknown_family"


def test_every_input_kind_gets_a_plain_name():
    assert names.input_name("log1p_n_want") == "Want to Read count (log)"
    assert names.input_name("svd_1") == "Topic 2: romance, contemporary; not fiction, children"
    assert names.input_name("subj_children_s_fiction") == "Subject flag: children's fiction"
    assert names.input_name("subj_man_woman_relationships_fiction") == "Subject flag: man-woman relationships fiction"
    assert names.input_name("subj_new_york_times_bestseller") == "Subject flag: New York Times bestseller"


def test_twenty_topics_numbered_from_one():
    assert len(names.TOPICS) == 20
    assert all(label.startswith(f"Topic {int(k.split('_')[1]) + 1}:") for k, label in names.TOPICS.items())


def test_arm_names():
    assert names.arm_setting("counts") == "Count columns"
    assert names.arm_value("counts", "log1p") == "log scale"
    assert names.arm_value("interactions", False) == "none"
    assert names.arm_value("target", "logit") == "log-odds"


def test_round_labels():
    assert names.round_label("r1", "64c7c902e4f2 (first 8)", 2) == "Round 1 (8 examples)"
    assert names.round_label("r1", "64c7c902e4f2 (all 22)", 2) == "Round 1 (all 22 examples)"
    assert names.round_label("r6", "3dd70bf5e831", 1) == "Round 6"


def test_verdict_kinds_and_checks():
    assert names.kind_name("control_rules") == "Planted: praise word and author name"
    assert names.check_names("faithful, rules_ok") == "accuracy, style rules"
    assert names.check_names("") == ""
