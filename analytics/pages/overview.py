"""Overview: what the model predicts, the core terms, the headline figures and a page guide."""

import pandas as pd
import streamlit as st

import cached
import load
import ui

metrics = cached.json_file("metrics_day5.json")
explain = cached.json_file("explain_day5.json")
llm = cached.json_file("llm_eval.json")

share = metrics[load.SHIPPED_TARGET]
app_model = share["families"][load.SHIPPED_FAMILY]
baseline = share["families"][load.BASELINE_FAMILY]
inputs = [e["name"] for e in explain["permutation"]]
n_topics = sum(name.startswith("svd_") for name in inputs)

st.title("Epilogue analytics")
st.markdown(
    "Epilogue estimates how likely a reader is to finish a book, from Open Library shelf data. "
    "For each book (a \"work\" in Open Library, which covers all its editions) the model predicts "
    "**share**: of everyone who put it on their Want to Read or Already Read shelf, the fraction on "
    "Already Read. share = already read / (already read + want to read). It is a rate for the book, "
    "not a prediction about one person; the app then adjusts it with the reader's own history "
    "(Models page, personal layer). An older target, ratio = already read / want to read, is kept "
    "for comparison; it has no upper limit."
)

st.header("Terms used on every page")
TERMS = [
    ("Book (work)", "Open Library's record for a book, covering all its editions."),
    ("Share (used by the app)", "The figure the app's model predicts: already read / (already read + want to read)."),
    ("Ratio (older target)", "Already read / want to read, kept for comparison. It has no upper limit."),
    ("Want to Read count", "How many people put the book on their Want to Read shelf."),
    ("Test folds", "The books are split five ways by first author. Each model is trained on four parts and "
                   "tested on the fifth, five times over, so it is always tested on authors it has not seen."),
    ("The app's model", "HistGB with tuned settings, trained on share. Tables mark it \"the app's model\" "
                        "or \"used by the app\"."),
]
ui.static_table(pd.DataFrame(TERMS, columns=["term", "meaning"]), labels={"term": "Term", "meaning": "Meaning"})

st.header("The model in the app")
ui.explain("mae", "r2")

tiles = st.columns(4, gap="medium")
with tiles[0]:
    st.metric(
        "Average error (MAE)",
        f"{app_model['mae']:.4f}",
        delta=f"{load.improvement(app_model['mae'], baseline['mae']):.0%} lower than always guessing the median "
        f"({baseline['mae']:.4f})",
        delta_color="off",
        delta_arrow="off",
        help=f"The predicted share is on average {app_model['mae']:.2f} away from the true share, "
        f"about {app_model['mae'] * 100:.0f} percentage points.",
    )
with tiles[1]:
    st.metric("R squared", f"{app_model['r2']:.4f}")
with tiles[2]:
    st.metric("Books in training", f"{share['n_rows']:,}")
with tiles[3]:
    st.metric(
        "Model inputs",
        f"{len(inputs)}",
        help=f"{len(inputs) - n_topics} book facts plus {n_topics} subject topics",
    )

latest = llm["rounds"][-1]
st.markdown(
    f"Spoiler-free summaries: in the latest round of prompt testing (round {latest['round'].removeprefix('r')}), "
    f"the main judge, Claude Opus, accepted {latest['judge_pass']} of the {latest['n']} summaries. "
    "The LLM evaluation page has the detail."
)

st.header("In this dashboard")
st.markdown(
    "- **Data**: where the data comes from, how the training books were chosen, how share is spread, "
    "missing values, which inputs move with share, and seasonal patterns.\n"
    f"- **Models**: how {len(share['families'])} models compare, why the app uses HistGB, where the model "
    "errs, and how a reader's own history adjusts it.\n"
    "- **Features**: the model's inputs, how the inputs are prepared before training, and what drives "
    "the prediction.\n"
    "- **LLM evaluation**: the summary prompt's rounds, the two judges' agreement and the "
    "spine-reading results.\n"
    "- **Stack**: the APIs and models the app calls, plus the offline summary judges and the "
    "model kept for comparison, which the app does not use."
)
