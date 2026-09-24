"""Build llm_eval.json, the summary-prompt evaluation the dashboard reads.

Round metrics come from summary-eval-metrics.csv; judge figures for rounds 4
to 6 are computed from the verdict files and checked against summary-eval.md.

Run: python analytics/build_llm_eval.py --review-dir <evaluation folder> --out <data>/processed/llm_eval.json
"""

import argparse
import csv
import json
import os
import re
import statistics
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

OUT_NAME = "llm_eval.json"

JUDGED_ROUNDS = ("r4", "r5", "r6")
JUDGES = ("opus", "sonnet")
CONTENT_CRITERIA = ("faithful", "no_outcome", "premise_kept", "null_correct")
# Each planted control carries one known error; it is caught when the judge
# fails it on this criterion.
CONTROL_CRITERION = {
    "control_invented": "faithful",
    "control_outcome": "no_outcome",
    "control_premise": "premise_kept",
    "control_rules": "rules_ok",
}
HOOK_SCORE = {"yes": 1.0, "partly": 0.5, "no": 0.0}

# summary-eval.md table rows: measure pattern -> (figure key, pair figure?).
MD_MEASURES = [
    (r"Real replies passing", "real_pass", False),
    (r"Real replies correct on content", "real_content_pass", False),
    (r"Duke (book, \d+ )?repeats correct on content", "repeats_content_pass", False),
    (r"Hook score", "hook_score", False),
    (r"Planted controls caught", "controls_caught", False),
    (r"Pass or fail agreement", "pass_fail_agreement", True),
    (r"Cohen's kappa", "cohens_kappa", True),
    (r"Content agreement", "content_agreement", True),
]
KAPPA_OF = {"pass_fail_agreement": "cohens_kappa", "content_agreement": "content_kappa"}
# The "of N" in a per-judge "X of N" cell is this figure's denominator.
DENOMINATOR_OF = {
    "real_pass": "real_n",
    "real_content_pass": "real_n",
    "repeats_content_pass": "repeats_n",
    "controls_caught": "controls_n",
}
CSV_INT = {"judge_pass", "n", "n_replies", "null_replies", "max_words", "over_60_words", "title_leaks", "praise_word_hits"}


def read_rounds_csv(path: Path) -> list[dict]:
    """Per-round rows of summary-eval-metrics.csv, numbers parsed.

    :param path: the CSV file.
    :return: one dict per row, with a source naming the file line.
    """
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            parsed: dict = {}
            for key, value in row.items():
                if key in CSV_INT:
                    parsed[key] = int(value)
                elif key in {"round", "variant", "prompt_version"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            parsed["source"] = f"{path.name}, line {line_no}"
            rows.append(parsed)
    return rows


def run_file_name(rnd: str) -> str:
    """The raw run file for a round: summary-eval-raw.json for r1, else summary-eval-raw-{round}.json.

    :param rnd: round id, e.g. "r2".
    :return: the file name.
    """
    return "summary-eval-raw.json" if rnd == "r1" else f"summary-eval-raw-{rnd}.json"


def run_hash_differs(csv_value: str, run_value: str | None) -> bool:
    """Whether a round's run file stamps a different prompt hash from the CSV.

    The CSV may add a variant note after the hash ("64c7c902e4f2 (first
    8)"); only the hash is compared.

    :param csv_value: summary-eval-metrics.csv prompt_version.
    :param run_value: the run file's prompt_version, or None if no file.
    :return: True when both exist and the hashes differ.
    """
    return run_value is not None and csv_value.split()[0] != run_value


def lines_with(lines: Sequence[str], text: str) -> str:
    """1-based line numbers of a file's lines that contain text, as prose.

    :param lines: the file's lines.
    :param text: the string to look for.
    :return: e.g. "lines 63, 65, 136", or "no line".
    """
    found = [str(i + 1) for i, line in enumerate(lines) if text in line]
    return f"line{'s' if len(found) > 1 else ''} {', '.join(found)}" if found else "no line"


def record_prompt_versions(rows: list[dict], folder: Path, md_lines: Sequence[str]) -> list[dict]:
    """Keep each round's CSV prompt hash and record its run file's hash beside it.

    The runner hashes the whole prompt module (all 22 examples) whatever subset a round
    sent, so a differing run-file hash names the same prompt; the CSV's is the one sent.

    :param rows: read_rounds_csv output.
    :param folder: the evaluation folder.
    :param md_lines: summary-eval.md lines.
    :return: the rows, each with prompt_version_run_file and, where the
        hashes differ, prompt_version_run and prompt_version_note.
    """
    prompt_md = folder / "summary-prompt.md"
    prompt_lines = prompt_md.read_text(encoding="utf-8").splitlines() if prompt_md.exists() else []
    for row in rows:
        path = folder / run_file_name(row["round"])
        run_value = json.loads(path.read_text(encoding="utf-8")).get("prompt_version") if path.exists() else None
        row["prompt_version_run_file"] = path.name if path.exists() else None
        if run_hash_differs(row["prompt_version"], run_value):
            csv_hash = row["prompt_version"].split()[0]
            row["prompt_version_run"] = run_value
            row["prompt_version_note"] = (
                f"Two hashes of the same prompt. {csv_hash} (summary-eval-metrics.csv) hashes what the round sent; "
                f"summary-prompt.md names it at {lines_with(prompt_lines, csv_hash)}. {run_value} ({path.name}) is "
                f"the runner's hash of the whole prompt module, the instruction with all 22 examples; "
                f"summary-eval.md explains it at {lines_with(md_lines, run_value)}."
            )
    return rows


def content_ok(verdict: Mapping) -> bool:
    """Whether a judge's verdict passes all four content criteria.

    :param verdict: one item from a summary-judge-*.json list.
    :return: True when faithful, no_outcome, premise_kept and null_correct all hold.
    """
    return all(verdict[c] for c in CONTENT_CRITERIA)


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Cohen's kappa for two raters' binary labels on the same items.

    :param a: first rater's labels.
    :param b: second rater's labels, same order.
    :return: kappa, or None when chance agreement is already 1.
    """
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def judge_figures(verdicts: Sequence[Mapping], key: Mapping[str, Mapping]) -> dict:
    """One judge's figures for one round.

    :param verdicts: the judge's list of item verdicts.
    :param key: the round's batch key, id -> {kind, work_key}.
    :return: real_pass, real_n, real_content_pass, controls_caught,
        controls_n, hook_score, repeats_content_pass, repeats_n.
    """
    by_kind: dict[str, list[Mapping]] = {}
    for v in verdicts:
        kind = key[v["id"]]["kind"]
        by_kind.setdefault("control" if kind.startswith("control_") else kind, []).append(v)
    real = by_kind.get("real", [])
    controls = by_kind.get("control", [])
    repeats = by_kind.get("repeat", [])
    return {
        "real_pass": sum(v["verdict"] == "PASS" for v in real),
        "real_n": len(real),
        "real_content_pass": sum(content_ok(v) for v in real),
        "controls_caught": sum(not v[CONTROL_CRITERION[key[v["id"]]["kind"]]] for v in controls),
        "controls_n": len(controls),
        "hook_score": statistics.mean(HOOK_SCORE[v["hook"]] for v in real),
        "repeats_content_pass": sum(content_ok(v) for v in repeats),
        "repeats_n": len(repeats),
    }


def pair_figures(opus: Sequence[Mapping], sonnet: Sequence[Mapping]) -> dict:
    """Agreement between the two judges over every item in the batch.

    :param opus: Opus's verdicts.
    :param sonnet: Sonnet's verdicts.
    :return: pass_fail_agreement, cohens_kappa, content_agreement,
        content_kappa, n_items.
    """
    b_by_id = {v["id"]: v for v in sonnet}
    ids = [v["id"] for v in opus if v["id"] in b_by_id]
    a_by_id = {v["id"]: v for v in opus}
    pass_a = [a_by_id[i]["verdict"] == "PASS" for i in ids]
    pass_b = [b_by_id[i]["verdict"] == "PASS" for i in ids]
    cont_a = [content_ok(a_by_id[i]) for i in ids]
    cont_b = [content_ok(b_by_id[i]) for i in ids]
    return {
        "pass_fail_agreement": sum(x == y for x, y in zip(pass_a, pass_b)) / len(ids),
        "cohens_kappa": cohens_kappa(pass_a, pass_b),
        "content_agreement": sum(x == y for x, y in zip(cont_a, cont_b)) / len(ids),
        "content_kappa": cohens_kappa(cont_a, cont_b),
        "n_items": len(ids),
    }


def _numbers(cell: str) -> list[str]:
    """Decimal or integer tokens in a markdown cell, in order."""
    return re.findall(r"\d+(?:\.\d+)?", cell)


def parse_md_round(lines: Sequence[str], round_no: int) -> dict:
    """Figures from one round's Opus/Sonnet table in summary-eval.md.

    :param lines: the file's lines.
    :param round_no: 4, 5 or 6.
    :return: {"opus": {...}, "sonnet": {...}, "pair": {...}}, each figure
        as {"text": the cell's number as written, "line": 1-based line}.
    """
    out: dict = {"opus": {}, "sonnet": {}, "pair": {}}
    start = next(i for i, s in enumerate(lines) if s.startswith(f"## Round {round_no} "))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    for i in range(start, end):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        for prefix, fig, is_pair in MD_MEASURES:
            if not re.match(prefix, cells[0]):
                continue
            items = re.search(r"(\d+) items", cells[0])
            if items:
                out["pair"]["n_items"] = {"text": items.group(1), "line": i + 1}
            if is_pair:
                nums = _numbers(cells[1].split("kappa")[0])
                # A "N items" count in the measure name is not the figure.
                if nums:
                    out["pair"][fig] = {"text": nums[0], "line": i + 1}
                if "kappa" in cells[1] and fig in KAPPA_OF:
                    out["pair"][KAPPA_OF[fig]] = {"text": _numbers(cells[1].split("kappa")[1])[0], "line": i + 1}
            else:
                for judge, cell in zip(JUDGES, cells[1:3]):
                    # "same" repeats the Opus cell for Sonnet.
                    cell = cells[1] if cell.lower() == "same" else cell
                    nums = _numbers(cell)
                    if nums:
                        out[judge][fig] = {"text": nums[0], "line": i + 1}
                    of = re.match(r"\s*\d+ of (\d+)", cell)
                    if of and fig in DENOMINATOR_OF:
                        out[judge][DENOMINATOR_OF[fig]] = {"text": of.group(1), "line": i + 1}
    return out


def matches(value: float | int | None, text: str) -> bool:
    """Whether a computed figure equals the markdown figure at its precision.

    :param value: the computed figure.
    :param text: the number as written, e.g. "0.83" or "10".
    :return: True when value rounds to text.
    """
    if value is None:
        return False
    places = len(text.split(".")[1]) if "." in text else 0
    return round(float(value), places) == round(float(text), places)


def figure(value, md: Mapping | None, route: str, source: str) -> dict:
    """One figure with its route, source and markdown check.

    :param value: the figure.
    :param md: the parsed markdown cell, or None when the table lacks it.
    :param route: "computed" or "transcribed".
    :param source: where the value came from.
    :return: {value, route, source, md_text, md_line, md_matches}.
    """
    return {
        "value": value,
        "route": route,
        "source": source,
        "md_text": md["text"] if md else None,
        "md_line": md["line"] if md else None,
        "md_matches": matches(value, md["text"]) if md else None,
    }


def build_round(folder: Path, rnd: str, md_lines: Sequence[str]) -> dict:
    """Judge figures for one round, computed where possible.

    :param folder: the evaluation folder.
    :param rnd: "r4", "r5" or "r6".
    :param md_lines: summary-eval.md lines.
    :return: {round, per_judge: {judge: {fig: figure}}, agreement: {fig: figure}}.
    """
    md = parse_md_round(md_lines, int(rnd[1:]))
    key_name = f"summary-judge-key-{rnd}.json"
    try:
        key = json.loads((folder / key_name).read_text(encoding="utf-8"))
        verdicts = {
            j: json.loads((folder / f"summary-judge-{rnd}-{j}.json").read_text(encoding="utf-8")) for j in JUDGES
        }
        per_judge = {j: judge_figures(verdicts[j], key) for j in JUDGES}
        pair = pair_figures(verdicts["opus"], verdicts["sonnet"])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"{rnd}: judge files unusable ({exc!r}); transcribing from summary-eval.md")
        return transcribed_round(rnd, md)
    return {
        "round": rnd,
        "per_judge": {
            j: {
                k: figure(v, md[j].get(k), "computed", f"summary-judge-{rnd}-{j}.json with {key_name}")
                for k, v in per_judge[j].items()
            }
            for j in JUDGES
        },
        "agreement": {
            k: figure(v, md["pair"].get(k), "computed", f"summary-judge-{rnd}-opus.json and -sonnet.json with {key_name}")
            for k, v in pair.items()
        },
    }


CRITERIA = ("faithful", "no_outcome", "premise_kept", "null_correct", "rules_ok")


def item_verdicts(
    verdicts: Mapping[str, Sequence[Mapping]], key: Mapping[str, Mapping], titles: Mapping[str, str]
) -> list[dict]:
    """Each batch item with both judges' verdicts, failed criteria and hook.

    :param verdicts: judge name -> that judge's verdict list.
    :param key: the round's batch key, id -> {kind, work_key}.
    :param titles: work_key -> title, from the round's raw run file.
    :return: one dict per item, in batch id order.
    """
    by_judge = {j: {v["id"]: v for v in vs} for j, vs in verdicts.items()}
    items = []
    for item_id in sorted(key):
        work_key = key[item_id]["work_key"]
        judges = {}
        for judge, by_id in by_judge.items():
            v = by_id.get(item_id)
            if v is not None:
                failed = [c for c in CRITERIA if not v[c]]
                judges[judge] = {"verdict": v["verdict"], "failed": failed, "hook": v.get("hook")}
        items.append({"id": item_id, "kind": key[item_id]["kind"], "work_key": work_key,
                      "title": titles.get(work_key), "judges": judges})
    return items


def build_verdicts(folder: Path, rnd: str) -> dict | None:
    """Per-book verdicts for one judged round, or None when its files are missing.

    :param folder: the evaluation folder.
    :param rnd: "r4", "r5" or "r6".
    :return: {round, source, items}.
    """
    key_name, raw_name = f"summary-judge-key-{rnd}.json", run_file_name(rnd)
    try:
        key = json.loads((folder / key_name).read_text(encoding="utf-8"))
        verdicts = {
            j: json.loads((folder / f"summary-judge-{rnd}-{j}.json").read_text(encoding="utf-8")) for j in JUDGES
        }
        raw = json.loads((folder / raw_name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"{rnd}: no per-book verdicts ({exc!r})")
        return None
    titles = {b["work_key"]: b.get("title") for b in raw.get("books", [])}
    return {
        "round": rnd,
        "source": f"summary-judge-{rnd}-opus.json and -sonnet.json with {key_name}; titles from {raw_name}",
        "items": item_verdicts(verdicts, key, titles),
    }


def transcribed_round(rnd: str, md: Mapping) -> dict:
    """Fallback: a round's figures as written in summary-eval.md.

    :param rnd: round id.
    :param md: parse_md_round output.
    :return: same shape as build_round, route "transcribed".
    """
    def fig(cell: Mapping) -> dict:
        return figure(float(cell["text"]), cell, "transcribed", f"summary-eval.md, line {cell['line']}")

    return {
        "round": rnd,
        "per_judge": {j: {k: fig(c) for k, c in md[j].items()} for j in JUDGES},
        "agreement": {k: fig(c) for k, c in md["pair"].items()},
    }


def build(folder: Path, out: Path) -> Path:
    """Write llm_eval.json and return its path.

    :param folder: the evaluation folder.
    :param out: the file to write.
    :return: out.
    """
    md_lines = (folder / "summary-eval.md").read_text(encoding="utf-8").splitlines()
    rounds = record_prompt_versions(read_rounds_csv(folder / "summary-eval-metrics.csv"), folder, md_lines)
    judges = [build_round(folder, r, md_lines) for r in JUDGED_ROUNDS]
    payload = {
        "built": date.today().isoformat(),
        "built_by": "analytics/build_llm_eval.py",
        "sources": {
            "rounds": "summary-eval-metrics.csv (prompt_version as written); each round's run file "
            "hash kept beside it as prompt_version_run where it differs, see prompt_version_note",
            "judges": "summary-judge-r{4,5,6}-{opus,sonnet}.json with summary-judge-key-r{4,5,6}.json",
            "check": "summary-eval.md, round tables (Round 4, 5 and 6 sections)",
        },
        "definitions": {
            "content_ok": "faithful, no_outcome, premise_kept and null_correct all true",
            "controls_caught": "planted control failed on its own criterion: " + json.dumps(CONTROL_CRITERION),
            "hook_score": "mean over real replies of yes 1, partly 0.5, no 0",
            "agreement": "share of all batch items (real, repeats, controls) where the two judges agree",
        },
        "rounds": rounds,
        "judges": judges,
        "verdicts": [v for v in (build_verdicts(folder, r) for r in JUDGED_ROUNDS) if v is not None],
        "verdicts_note": (
            "Per-book verdicts come from the judge files, which exist for rounds 4 to 6 only; rounds 1 to 3 "
            "were judged in the written evaluation only (summary-eval.md verdict tables) and are not included."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def main() -> None:
    """Command line: build from --review-dir into --out, and print the check against the write-up."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review-dir", type=Path, required=True, help="folder holding the summary-eval files")
    parser.add_argument("--out", type=Path, help="file to write; defaults to $CAPSTONE_DATA/processed/" + OUT_NAME)
    args = parser.parse_args()
    out = args.out
    if out is None:
        if not os.getenv("CAPSTONE_DATA"):
            parser.error("give --out or set CAPSTONE_DATA")
        out = Path(os.environ["CAPSTONE_DATA"]) / "processed" / OUT_NAME
    path = build(args.review_dir, out)
    data = json.loads(path.read_text(encoding="utf-8"))
    for r in data["judges"]:
        figs = [(j, k, f) for j, d in r["per_judge"].items() for k, f in d.items()]
        figs += [("pair", k, f) for k, f in r["agreement"].items()]
        checked = [f for _, _, f in figs if f["md_text"] is not None]
        bad = [(j, k, f["value"], f["md_text"]) for j, k, f in figs if f["md_matches"] is False]
        print(f"{r['round']}: {len(figs)} figures, {len(checked)} checked against summary-eval.md, mismatches {bad}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
