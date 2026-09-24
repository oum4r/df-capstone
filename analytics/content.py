"""Text the pages show that no data file holds: definitions, dataset and stack tables, caveats."""

# Plain-words definitions, shown before a measure is first used ("measure: meaning").
METRICS = {
    "mae": "MAE, mean absolute error: on average, how far off the prediction is, in the target's own units.",
    "mse": "MSE, mean squared error: the average of the squared misses, so big misses count much more.",
    "rmse": "RMSE, root mean squared error: the square root of MSE, back in target units but still punishing big misses.",
    "r2": "R squared: the share of the target's variation the model explains; 0 is no better than the mean, 1 is perfect.",
    "spearman": "Spearman correlation: how closely two figures rise and fall together in rank order, from -1 to +1.",
    "fold_std": (
        "Spread across folds (fold std): how much the MAE moves between the five test folds "
        "(the standard deviation of the five fold MAEs)."
    ),
    "kappa": (
        "Agreement beyond chance (Cohen's kappa): how often the two judges agree once chance agreement is "
        "taken out; 1 is full agreement, 0 is chance level."
    ),
    "controls": (
        "Planted tests: four summaries that passed in round 2, each given one known error (an invented fact, "
        "an ending given away, a dropped set-up, a praise word plus the author's name); a judge catches one by "
        "failing it for that exact error."
    ),
    "hook": (
        "Hook score: whether the summary keeps the book's hook, scored yes 1, partly 0.5, no 0 and averaged "
        "over the 12 real summaries."
    ),
    "compression": (
        "Compression: the summary's length in words as a share of the description's, averaged over the summaries."
    ),
    "rouge": (
        "ROUGE-L precision: the share of the summary's words that also appear, in the same order, in the "
        "description. Lower means more of it is in the model's own words."
    ),
    "log_loss": (
        "Log loss: how far the predicted chance of finishing is from what happened, punishing confident misses "
        "hardest; lower is better."
    ),
}

DATASETS = [
    {
        "dataset": "Open Library reading-log dump",
        "what": "Shelf events (book, edition, shelf, date), dated 2026-08-31; no user IDs",
        "size": "116 MB compressed, 12,838,026 rows, 3,314,590 books",
        "became": "Shelf counts per book, then share and ratio per book",
        "status": "used",
    },
    {
        "dataset": "Open Library works dump",
        "what": "One record per book: title, subjects, authors",
        "size": "4.1 GB compressed",
        "became": "Title, subjects and authors per book (67,350 of the 68,734 found)",
        "status": "used",
    },
    {
        "dataset": "Open Library editions dump",
        "what": "Every edition, streamed and filtered to the model's books",
        "size": "11.7 GB compressed, 56.7M editions",
        "became": "Median pages, earliest year, edition count and English share per book",
        "status": "used",
    },
    {
        "dataset": "StoryGraph export",
        "what": "A reader's own CSV (23 columns)",
        "size": "small",
        "became": "Personal finish rate by page length",
        "status": "used",
    },
    {
        "dataset": "Mock personas (synthetic)",
        "what": "Three made-up StoryGraph readers over real Open Library ISBNs",
        "size": "110, 87 and 60 rows",
        "became": "Stand-in history for the personal blend",
        "status": "used, synthetic",
    },
    {
        "dataset": "Profile study (synthetic)",
        "what": "200 simulated readers each reading 40 real books",
        "size": "8,000 reader-book rows",
        "became": "The profile feature study (Models page)",
        "status": "used, synthetic",
    },
    {
        "dataset": "Summary eval set",
        "what": "The same 12 Open Library books each round, summarised by Gemini 3.5 Flash-Lite",
        "size": "12 books; rounds 4 to 6 add 4 planted tests, rounds 5 and 6 repeats",
        "became": "The LLM evaluation page",
        "status": "used",
    },
    {
        "dataset": "DAIGT-V2 (Kaggle)",
        "what": "Human- and AI-written texts, labelled, for a planned text authenticity check; about 45K rows",
        "size": "about 50 MB",
        "became": "nothing yet",
        "status": "planned, not started",
    },
]

FUNNEL = [
    {"stage": "Books ever shelved", "works": 3_314_590},
    {"stage": "At least 5 Want to Read and 3 Already Read shelvings", "works": 68_734},
    {"stage": "Found in the works dump: the training books", "works": 67_350},
]

# The 15 base inputs by group; "svd" takes every subject topic.
FEATURE_GROUPS = [
    {"group": "Shelf counts (log scale)", "columns": ["log1p_n_want", "log1p_n_reading", "n_stopped"]},
    {"group": "Shelf age", "columns": ["shelf_age_days"]},
    {"group": "Number of subjects", "columns": ["n_subjects"]},
    {"group": "Number of authors", "columns": ["n_authors"]},
    {"group": "Has a description", "columns": ["has_description"]},
    {"group": "Number of covers", "columns": ["n_covers"]},
    {"group": "Publish year", "columns": ["first_publish_year", "first_publish_year_missing"]},
    {
        "group": "Editions",
        "columns": ["pages_median", "publish_year_min", "n_editions", "share_english", "pages_median_missing"],
    },
    {"group": "Subject topics", "columns": "svd"},
]

# What each input-preparation choice does to the stored columns before training.
ARM_NOTES = {
    ("counts", "log1p"): "Stopped Reading count, edition count, median pages and subject count put on a log scale",
    ("counts", "raw"): "Want to Read and Currently Reading counts used as plain counts",
    ("year", "raw"): "first publish year kept as a number",
    ("year", "era"): "first publish year replaced by four era columns",
    ("subjects", "svd20"): "the 60 subject flags replaced by 20 subject topics",
    ("missing", "native"): "gaps left in place for the model to handle",
    ("interactions", "False"): "no interaction columns added",
    ("scaling", "none"): "no scaling",
    ("weights", "none"): "every book weighted equally",
    ("weights", "log1p_n_want"): "books with more Want to Read shelvings count more in training",
    ("target", "logit"): "share kept between 0.001 and 0.999 and modelled on the log-odds scale",
    ("target", "log1p"): "modelled as the log of 1 plus ratio, then converted back",
    ("cat_subject", "True"): "the first subject added as a category",
}

APIS = [
    {
        "api": "Open Library",
        "endpoints": "openlibrary.org /isbn/{isbn13}.json, /search.json, /works/{id}.json, /works/{id}/editions.json",
        "use": "Book lookup by ISBN or title, work records, editions",
    },
    {"api": "Open Library Covers", "endpoints": "covers.openlibrary.org/b/id/{cover_id}-M.jpg", "use": "Cover images"},
    {
        "api": "Google Gemini API",
        "endpoints": "generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "use": "Spine reading (first choice) and spoiler-free summaries",
    },
    {"api": "OpenRouter", "endpoints": "openrouter.ai/api/v1/chat/completions", "use": "Spine reading fallback"},
]

LLMS = [
    {
        "model": "Gemini 3.5 Flash-Lite",
        "provider": "Google Gemini API",
        "role": "Spine reading (reading titles from a photo of a shelf), first choice; the only summary model",
    },
    {
        "model": "Gemini 3.6 Flash",
        "provider": "Google Gemini API",
        "role": "Spine reading, second Gemini choice; tried first when the user asks for a slower, more careful read",
    },
    {"model": "Gemma 4 31B (free tier, then paid)", "provider": "OpenRouter", "role": "Spine reading fallback"},
    {"model": "Gemma 3 12B", "provider": "OpenRouter", "role": "Spine reading, last fallback"},
    {
        "model": "Claude Opus and Claude Sonnet",
        "provider": "Anthropic, run offline by the project (not by the app)",
        "role": "Judges of the summary prompt, rounds 4 to 6",
    },
]

REGRESSORS = [
    {
        "model": "HistGB (scikit-learn's HistGradientBoostingRegressor)",
        "role": "Used in the app: tuned settings, trained on share on a log-odds scale",
    },
    {"model": "CatBoost", "role": "For comparison: the most accurate single model on both targets, not used"},
]

VISION_RESULTS = [
    {"model": "Gemma 4 31B via OpenRouter (free tier)", "result": "9 of 9 exact, when OpenRouter's free capacity is available"},
    {"model": "Gemma 4 31B via OpenRouter (paid)", "result": "same model, paid, about $0.0002 a shelf"},
    {"model": "Gemma 3 12B via OpenRouter", "result": "9 of 9 identifiable, one title merged"},
    {"model": "Gemini 3.5 Flash-Lite", "result": "9 of 9 spines read correctly in 3 s; free tier 15 requests a minute, 500 a day"},
    {"model": "Gemini 3.6 Flash", "result": "also 9 of 9; free tier 5 requests a minute, 20 a day"},
]

CAVEATS = {
    "synthetic_readers": (
        "These readers are synthetic personas, not real people. Their outcomes were drawn from a model "
        "that already contains each book's population share, so the learned weights reflect the "
        "generator, not real readers. Only a real reader's export will show whether the fitted weight helps."
    ),
    "profile_study": (
        "Every number in the profile study is synthetic: 200 simulated readers, and the feature ordering "
        "reflects the generator's own preference spreads, not evidence about real readers."
    ),
}
