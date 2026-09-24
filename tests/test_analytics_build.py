"""Tests for analytics/build_llm_eval.py and the static table helper in analytics/ui.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analytics"))

import build_llm_eval as build  # noqa: E402
import theme  # noqa: E402
import ui  # noqa: E402

MD = """# Eval

## Round 5 (something, prompt abc)

| Measure | Opus | Sonnet |
|---|---|---|
| Real replies correct on content | 11 of 12 | 11 of 12 |
| Duke book, 3 repeats correct on content | 1 of 3 | 1 of 3 |
| Duke book over all 4 draws | 2 correct | same |
| Planted controls caught | 4 of 4 | same |
| Hook score | 0.875 | 1.00 |
| Content agreement, 19 items | 0.947, kappa 0.883 | |

## Round 6 (next)

| Real replies correct on content | 9 of 12 | 9 of 12 |
""".splitlines()


# Round 2's run file (50c6dfd7b6cd, all 22 examples) and the CSV (5f56e4ab84a5, the 8
# examples sent) hash the same prompt two ways, so both are kept.


def test_run_hash_differs_compares_hashes_only():
    assert build.run_hash_differs("5f56e4ab84a5", "50c6dfd7b6cd")
    assert not build.run_hash_differs("64c7c902e4f2 (first 8)", "64c7c902e4f2")
    assert not build.run_hash_differs("72c8d8c81156", None)


def test_run_file_names():
    assert build.run_file_name("r1") == "summary-eval-raw.json"
    assert build.run_file_name("r2") == "summary-eval-raw-r2.json"


def test_record_prompt_versions_keeps_both_hashes(tmp_path):
    (tmp_path / "summary-eval-raw-r2.json").write_text('{"prompt_version": "50c6dfd7b6cd"}', encoding="utf-8")
    (tmp_path / "summary-prompt.md").write_text("title\n\nPrompt version 5f56e4ab84a5.\n", encoding="utf-8")
    rows = [{"round": "r2", "prompt_version": "5f56e4ab84a5"}, {"round": "r3", "prompt_version": "72c8d8c81156"}]
    out = build.record_prompt_versions(rows, tmp_path, ["Prompt 50c6dfd7b6cd is the hash with all 22 examples"])
    assert out[0]["prompt_version"] == "5f56e4ab84a5"  # the CSV value stays
    assert out[0]["prompt_version_run"] == "50c6dfd7b6cd"  # the run file's is kept beside it
    note = out[0]["prompt_version_note"]
    assert "5f56e4ab84a5" in note and "50c6dfd7b6cd" in note and "line 3" in note and "line 1" in note
    assert out[1]["prompt_version"] == "72c8d8c81156" and "prompt_version_run" not in out[1]
    assert out[1]["prompt_version_run_file"] is None


def test_parse_md_round_reads_repeats_totals_and_items():
    md = build.parse_md_round(MD, 5)
    assert md["opus"]["real_content_pass"]["text"] == "11" and md["opus"]["real_n"]["text"] == "12"
    assert md["opus"]["repeats_content_pass"] == {"text": "1", "line": 8}
    assert md["sonnet"]["repeats_n"]["text"] == "3"
    assert md["sonnet"]["controls_caught"]["text"] == "4"  # "same" repeats the Opus cell
    assert md["pair"]["n_items"]["text"] == "19"
    assert md["pair"]["content_kappa"]["text"] == "0.883"
    assert "real_content_pass" in md["opus"] and md["opus"]["real_content_pass"]["line"] == 7  # not round 6's row


def test_matches_uses_the_written_precision():
    assert build.matches(0.8333, "0.83") and build.matches(12, "12") and not build.matches(0.6, "0.673")


def test_cohens_kappa():
    assert build.cohens_kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert build.cohens_kappa([True, True], [True, True]) is None


def test_html_table_escapes_formats_and_marks_missing():
    df = pd.DataFrame({"name": ["<b>x</b>", "y"], "mae": [0.06034, None], "cost": [0.000984, float("nan")]})
    out = ui.html_table(df, labels={"mae": "MAE"}, formats={"mae": "{:.4f}", "cost": ui.dollars},
                        numeric=["mae"], highlight=[True, False])
    assert "&lt;b&gt;x&lt;/b&gt;" in out and "<b>x</b>" not in out
    assert '<td class="num">0.0603</td>' in out and "$0.000984" in out
    assert out.count("n/a") == 2 and "None" not in out and "nan" not in out
    assert out.count('class="shipped"') == 1 and "<th class=\"num\">MAE</th>" in out


def test_split_definition_keeps_wording():
    assert ui.split_definition("MAE, mean absolute error: on average, how far off.") == (
        "MAE, mean absolute error",
        "On average, how far off.",
    )


def test_item_verdicts_lists_false_criteria_and_titles():
    ok = {"faithful": True, "no_outcome": True, "premise_kept": True, "null_correct": True, "rules_ok": True}
    verdicts = {"opus": [{"id": "J02", **ok, "rules_ok": False, "verdict": "PASS", "hook": "yes"},
                         {"id": "J01", **ok, "verdict": "PASS", "hook": "partly"}]}
    key = {"J02": {"kind": "control_rules", "work_key": "OL2W"}, "J01": {"kind": "real", "work_key": "OL1W"}}
    items = build.item_verdicts(verdicts, key, {"OL1W": "Book one"})
    assert [i["id"] for i in items] == ["J01", "J02"]
    assert items[0]["title"] == "Book one" and items[1]["title"] is None
    assert items[1]["judges"]["opus"] == {"verdict": "PASS", "failed": ["rules_ok"], "hook": "yes"}


def test_series_colours_give_the_accent_only_to_the_shipped_model():
    colours = theme.series_colours(["median", "hist_gb", "catboost"], "hist_gb")
    assert colours[1] == theme.ACCENT and theme.ACCENT not in (colours[0], colours[2])
    assert theme.ACCENT not in theme.series_colours(["a", "b"], None)
    assert len(theme.series_colours([str(i) for i in range(12)], None)) == 12
