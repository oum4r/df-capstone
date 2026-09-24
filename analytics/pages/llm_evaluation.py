"""LLM evaluation: the summary prompt's rounds, the two judges, per-book verdicts and spine reading."""

import altair as alt
import pandas as pd
import streamlit as st

import cached
import content
import load
import names
import theme
import ui

# Measure the rounds chart can plot: column -> (label, number format, y scale from zero).
ROUND_METRICS = {
    "judge_pass": ("Pass count (of 12)", "d", True),
    "rouge_l_precision_vs_description": ("ROUGE-L precision", ".3f", False),
    "compression_ratio": ("Compression", ".3f", False),
    "mean_words": ("Mean summary length (words)", ".1f", False),
    "median_seconds": ("Median reply time (s)", ".2f", False),
    "mean_cost_usd": ("Mean cost per summary (USD)", "$.6f", False),
}
ROUND_NUMBERS = ["judge_pass", "n", "mean_words", "max_words", "over_60_words",
                 "rouge_l_precision_vs_description", "compression_ratio", "praise_word_hits",
                 "median_seconds", "mean_cost_usd"]
ROUND_LABELS = {"label": "Round", "judge_pass": "Pass", "n": "Books tested",
                "mean_words": "Mean summary length (words)", "max_words": "Longest summary (words)",
                "over_60_words": "Summaries over 60 words",
                "rouge_l_precision_vs_description": "ROUGE-L precision", "compression_ratio": "Compression",
                "praise_word_hits": "Praise words used (a style rule bans them)",
                "median_seconds": "Median reply time (s)", "mean_cost_usd": "Mean cost per summary"}
ROUND_FORMATS = {"mean_words": "{:.1f}", "rouge_l_precision_vs_description": "{:.3f}",
                 "compression_ratio": "{:.3f}", "median_seconds": "{:.2f}", "mean_cost_usd": ui.dollars}
VERDICTS = {"PASS": "Pass", "FAIL": "Fail"}

llm = cached.json_file("llm_eval.json")
rounds = load.llm_rounds_table(llm)
judges = load.llm_judge_table(llm)
notes = ui.Notes()


def round_name(rnd: str) -> str:
    """Plain name of a judged round, e.g. "Round 4"."""
    return f"Round {rnd.removeprefix('r')}"


st.title("LLM evaluation")
st.markdown(
    "The book page offers a spoiler-free summary written by a language model from the Open Library "
    "description. The prompt was revised over six rounds and checked on the same 12 books each time."
)

st.header("Rounds")
ui.explain("rouge", "compression")
st.markdown(
    "Pass is how many of the 12 summaries the main judge, Claude Opus, accepted. Rounds 1 to 3 had one judge, "
    "Opus; from round 4 a second judge, Claude Sonnet, scored the same summaries. Round 1 tried two versions "
    "of the prompt: one with 8 worked examples and one with all 22."
)
c1, _ = st.columns([1, 2], gap="large")
with c1:
    metric = st.selectbox(
        "Measure across rounds", list(ROUND_METRICS), format_func=lambda m: ROUND_METRICS[m][0], index=0,
        key="llm_metric",
    )
label, fmt, zero = ROUND_METRICS[metric]
left, right = st.columns([3, 2], gap="large")
with left:
    chart = alt.Chart(rounds).mark_line(point=True, color=theme.NEUTRAL_FILL).encode(
        x=alt.X("label:N", sort=list(rounds["label"]), title="Round", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y(f"{metric}:Q", title=label, scale=alt.Scale(zero=zero)),
        tooltip=[alt.Tooltip("label:N", title="Round"), alt.Tooltip(f"{metric}:Q", title=label, format=fmt)],
    ).properties(height=340)
    theme.show_chart(chart)
with right:
    ui.static_table(
        rounds,
        order=["label", metric],
        labels={**ROUND_LABELS, metric: label},
        formats=ROUND_FORMATS,
        numeric=[metric],
    )

st.subheader("All measures by round")
ui.static_table(
    rounds,
    order=["label", *ROUND_NUMBERS],
    labels=ROUND_LABELS,
    formats=ROUND_FORMATS,
    numeric=ROUND_NUMBERS,
)
r5 = rounds.loc[rounds["round"] == "r5"].iloc[0]
notes.add(
    "Rounds",
    "Word counts split hyphenated words, as the evaluation script counts them. So in round 5 the longest summary "
    f"is {int(r5['max_words'])} words here and {int(r5['over_60_words'])} summaries run over 60 words. The written "
    "evaluation counts by spaces and finds 67 words and one summary over 60.",
)

st.header("Two blind judges, rounds 4 to 6")
ui.explain("kappa", "controls", "hook")
st.markdown(
    "From round 4, two Claude models, Opus and Sonnet, each scored the round's summaries with no record of "
    "earlier rounds and without being told which items were planted tests. A summary is correct on content "
    "when it is accurate, gives no ending away, keeps the set-up, and is left blank only when the description "
    "has nothing to summarise."
)
per_judge = judges.assign(
    real=lambda d: d["real_pass"].astype(int).astype(str) + " of " + d["real_n"].astype(int).astype(str),
    content=lambda d: d["real_content_pass"].astype(int).astype(str) + " of " + d["real_n"].astype(int).astype(str),
    caught=lambda d: d["controls_caught"].astype(int).astype(str) + " of " + d["controls_n"].astype(int).astype(str),
    judge=lambda d: d["judge"].map(names.JUDGES),
    round=lambda d: d["round"].map(round_name),
)
pairs = judges.drop_duplicates("round").assign(round=lambda d: d["round"].map(round_name))
AGREEMENT = ["pass_fail_agreement", "cohens_kappa", "content_agreement", "content_kappa"]
left, right = st.columns(2, gap="large")
with left:
    ui.static_table(
        per_judge,
        order=["round", "judge", "real", "content", "caught", "hook_score"],
        labels={"round": "Round", "judge": "Judge", "real": "Real summaries passing",
                "content": "Correct on content", "caught": "Planted tests caught", "hook_score": "Hook score"},
        formats={"hook_score": "{:.3f}"},
        numeric=["real", "content", "caught", "hook_score"],
    )
with right:
    ui.static_table(
        pairs,
        order=["round", "n_items", *AGREEMENT],
        labels={"round": "Round", "n_items": "Items judged", "pass_fail_agreement": "Pass or fail agreement",
                "cohens_kappa": "Agreement beyond chance (kappa)", "content_agreement": "Content agreement",
                "content_kappa": "Content agreement beyond chance (kappa)"},
        formats={"n_items": "{:,}", **{c: "{:.3f}" for c in AGREEMENT}},
        numeric=["n_items", *AGREEMENT],
    )
    st.markdown(
        "Agreement is the share of all items in a round (real summaries, repeats and planted tests) on which "
        "the two judges gave the same verdict: first the overall pass or fail, then the content verdict alone."
    )

routes = [f for r in llm["judges"] for d in [*r["per_judge"].values(), r["agreement"]] for f in d.values()]
n_computed = sum(f["route"] == "computed" for f in routes)
n_checked = sum(f["md_text"] is not None for f in routes)
n_match = sum(bool(f["md_matches"]) for f in routes)
notes.add(
    "Two blind judges",
    (f"All {len(routes)} judge figures in the build" if n_computed == len(routes)
     else f"{n_computed} of the {len(routes)} judge figures in the build")
    + " were computed from the judges' own score files. "
    + (f"The {n_checked} that the written evaluation also reports all agree with it."
       if n_match == n_checked else f"Of the {n_checked} that the written evaluation also reports, {n_match} agree with it."),
)

st.header("Per-book verdicts")
verdict_rounds = [v["round"] for v in llm.get("verdicts", [])]
if not verdict_rounds:
    st.markdown("No per-book verdicts were built for this dashboard.")
else:
    c1, _ = st.columns([1, 2], gap="large")
    with c1:
        rnd = st.selectbox(
            "Round", verdict_rounds, index=len(verdict_rounds) - 1, format_func=round_name, key="llm_round"
        )
    block = next(v for v in llm["verdicts"] if v["round"] == rnd)
    verdicts = load.llm_verdicts_table(llm, rnd)
    kinds = verdicts["kind"].map(lambda k: "control" if k.startswith("control_") else k).value_counts()
    parts = [f"{kinds.get('real', 0)} real summaries"]
    if kinds.get("repeat", 0):
        parts.append(f"{kinds['repeat']} repeats")
    parts.append(f"{kinds.get('control', 0)} planted tests")
    repeated = verdicts.loc[verdicts["kind"] == "repeat", "title"].dropna().unique()
    repeat_line = (
        f" A repeat asks for the same book again: this round asked for {repeated[0]} {kinds['repeat']} more "
        "times, to see whether a pass is stable."
        if kinds.get("repeat", 0) and len(repeated) == 1 else ""
    )
    st.markdown(
        f"Each item the two judges scored in {round_name(rnd).lower()}: " + ", ".join(parts[:-1])
        + f" and {parts[-1]}.{repeat_line} The failed-checks columns list what each judge said the summary got "
        "wrong. Under the judges' rules a style slip alone, such as a one-sentence summary, does not fail it, so a "
        "pass can still list \"style rules\". Rounds 1 to 3 were judged in the written evaluation only and are "
        "not included."
    )
    shown = verdicts.assign(
        kind=verdicts["kind"].map(names.kind_name),
        **{f"{j}_verdict": verdicts[f"{j}_verdict"].replace(VERDICTS) for j in names.JUDGES},
        **{f"{j}_failed": verdicts[f"{j}_failed"].map(names.check_names) for j in names.JUDGES},
    )
    ui.static_table(
        shown,
        order=["kind", "title", "opus_verdict", "opus_failed", "opus_hook",
               "sonnet_verdict", "sonnet_failed", "sonnet_hook"],
        labels={"kind": "Kind", "title": "Book", "opus_verdict": "Opus", "opus_failed": "Opus: checks failed",
                "opus_hook": "Opus: hook", "sonnet_verdict": "Sonnet", "sonnet_failed": "Sonnet: checks failed",
                "sonnet_hook": "Sonnet: hook"},
    )

st.header("Reading book spines from a photo")
st.markdown(
    "The shelf scan sends a photo to a vision model and asks for every title and author it can see. "
    "These results are reported in code comments; no results file backs them."
)
ui.static_table(
    pd.DataFrame(content.VISION_RESULTS),
    labels={"model": "Model", "result": "Reported result"},
)

notes.render()
