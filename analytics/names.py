"""Display names: one mapping from code names to the words a reader sees.

Pages show these names in options, table cells, axis labels, legends and
tooltips, never the code keys.
"""

import re
from collections.abc import Mapping

APP_TARGET = "share"
APP_FAMILY = "hist_gb"

FAMILIES = {
    "hist_gb": "HistGB (tuned)",
    "hist_gb_default": "HistGB (default settings)",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "gradient_boosting": "Gradient boosting (classic)",
    "bagging": "Bagged trees",
    "adaboost": "AdaBoost",
    "decision_tree": "Decision tree",
    "mlp": "Neural network (MLP)",
    "ridge": "Ridge regression",
    "lasso": "Lasso regression",
    "elastic_net": "Elastic net regression",
    "huber": "Huber regression",
    "linear_svr": "Linear SVR",
    "svr_rbf": "RBF SVR (10,000-row sample)",
    "knn": "Nearest neighbours (KNN)",
    "median": "Median guess (baseline)",
    "mean": "Mean guess (baseline)",
}

TARGETS = {"share": "Share (used by the app)", "ratio": "Ratio (older target)"}
TARGET_SHORT = {"share": "Share", "ratio": "Ratio"}
TARGET_FORMULAS = {
    "share": "already read / (already read + want to read)",
    "ratio": "already read / want to read",
}

INPUTS = {
    "share": "Share",
    "ratio": "Ratio",
    "log1p_n_want": "Want to Read count (log)",
    "log1p_n_reading": "Currently Reading count (log)",
    "n_stopped": "Stopped Reading count",
    "shelf_age_days": "Days since first shelved",
    "n_subjects": "Number of subjects",
    "n_authors": "Number of authors",
    "has_description": "Has a description",
    "n_covers": "Number of covers",
    "first_publish_year": "First publish year",
    "first_publish_year_missing": "First publish year missing",
    "pages_median": "Median page count",
    "pages_median_missing": "Page count missing",
    "publish_year_min": "Earliest edition year",
    "n_editions": "Number of editions",
    "share_english": "Share of editions in English",
}

# Subject topics, numbered from 1; words after "not" carry a minus sign in
# explain_day5.json svd_labels, meaning they pull the topic the other way.
TOPICS = {
    "svd_0": "Topic 1: fiction, romance, general, historical",
    "svd_1": "Topic 2: romance, contemporary; not fiction, children",
    "svd_2": "Topic 3: general; not contemporary, stories, and",
    "svd_3": "Topic 4: historical, regency, customs; not contemporary",
    "svd_4": "Topic 5: large type books, mystery",
    "svd_5": "Topic 6: man-woman relationships, large type",
    "svd_6": "Topic 7: mystery, detective, man-woman",
    "svd_7": "Topic 8: English, in; not New York",
    "svd_8": "Topic 9: English, American literature; not fantasy",
    "svd_9": "Topic 10: science; not children, juvenile, stories",
    "svd_10": "Topic 11: fantasy, American, stories; not fiction",
    "svd_11": "Topic 12: English, in, fantasy; not science",
    "svd_12": "Topic 13: fantasy, customs, contemporary",
    "svd_13": "Topic 14: graphic novels, comics; not fantasy",
    "svd_14": "Topic 15: contemporary, regency, horror; not romance",
    "svd_15": "Topic 16: American and juvenile literature; not stories",
    "svd_16": "Topic 17: horror, suspense, women; not science",
    "svd_17": "Topic 18: history, war, 1945; not regency",
    "svd_18": "Topic 19: horror, tales, literature; not love",
    "svd_19": "Topic 20: reading level, grade; not fictitious",
}

# Word fixes for subject flag names, applied before underscores become spaces.
_SUBJECT_FIXES = [
    ("children_s", "children's"),
    ("man_woman", "man-woman"),
    ("new_york_times", "New York Times"),
    ("new_york", "New York"),
    ("english", "English"),
    ("american", "American"),
    ("england", "England"),
    ("spanish", "Spanish"),
]

ARM_SETTINGS = {
    "counts": "Count columns",
    "year": "Publish year",
    "subjects": "Subjects",
    "missing": "Missing values",
    "interactions": "Interaction columns",
    "scaling": "Scaling",
    "weights": "Row weights",
    "target": "Target scale",
    "cat_subject": "First subject as a category",
}
ARM_VALUES = {
    ("counts", "log1p"): "log scale",
    ("counts", "raw"): "plain counts",
    ("year", "raw"): "as a number",
    ("year", "era"): "as four eras",
    ("subjects", "svd20"): "20 subject topics",
    ("missing", "native"): "left for the model",
    ("interactions", "False"): "none",
    ("scaling", "none"): "none",
    ("weights", "none"): "equal",
    ("weights", "log1p_n_want"): "by Want to Read count",
    ("target", "logit"): "log-odds",
    ("target", "log1p"): "log of 1 plus ratio",
    ("cat_subject", "True"): "yes",
}

PERSONAS = {
    "fast_fiction_reader": "Fast fiction reader",
    "serial_abandoner": "Serial abandoner",
    "slow_nonfiction_reader": "Slow non-fiction reader",
}

PROFILE_FEATURES = {
    "reader_finish_rate": "Reader's overall finish rate",
    "reader_abandon_rate": "Reader's abandon rate (1 minus finish rate)",
    "reader_band_rate": "Reader's finish rate at this page length",
    "reader_subject_rate": "Reader's finish rate in this subject group",
    "reader_format_rate": "Reader's finish rate in this format",
    "reader_band_lift": "Page-length rate minus overall rate",
    "reader_subject_lift": "Subject rate minus overall rate",
    "reader_format_lift": "Format rate minus overall rate",
    "length_gap": "Gap from the reader's usual length (hundreds of pages)",
    "books_per_month": "Books finished per month",
}

KINDS = {
    "real": "Real",
    "repeat": "Repeat (same book asked again)",
    "control_invented": "Planted: invented fact",
    "control_outcome": "Planted: ending given away",
    "control_premise": "Planted: dropped set-up",
    "control_rules": "Planted: praise word and author name",
}

CHECKS = {
    "faithful": "accuracy",
    "no_outcome": "ending",
    "premise_kept": "set-up",
    "null_correct": "blank reply",
    "rules_ok": "style rules",
}

JUDGES = {"opus": "Claude Opus", "sonnet": "Claude Sonnet"}


def family_name(key: str, target: str = APP_TARGET) -> str:
    """A model family's display name; the tuned HistGB is marked as the app's model on share.

    :param key: family key from metrics_day5.json.
    :param target: the target the name is shown for.
    :return: the display name, or the key itself when unmapped.
    """
    name = FAMILIES.get(key, key)
    return f"{name}, the app's model" if key == APP_FAMILY and target == APP_TARGET else name


def subject_flag_name(key: str) -> str:
    """A stored subject flag, named from its column slug.

    :param key: e.g. "subj_fiction_romance_historical_regency".
    :return: e.g. "Subject flag: fiction romance historical regency".
    """
    words = key.removeprefix("subj_")
    for old, new in _SUBJECT_FIXES:
        words = re.sub(rf"(?<![a-z]){old}(?![a-z])", new, words)
    return "Subject flag: " + words.replace("_", " ")


def input_name(key: str) -> str:
    """Any input, target, subject topic or subject flag, in plain words.

    :param key: a column name.
    :return: its display name, or the key itself when unmapped.
    """
    if key in INPUTS:
        return INPUTS[key]
    if key in TOPICS:
        return TOPICS[key]
    if key.startswith("subj_"):
        return subject_flag_name(key)
    return key


def arm_setting(setting: str) -> str:
    """Plain name of one input-preparation setting.

    :param setting: arm key, e.g. "counts".
    :return: e.g. "Count columns".
    """
    return ARM_SETTINGS.get(setting, setting)


def arm_value(setting: str, value: object) -> str:
    """The words shown for one input-preparation value.

    :param setting: arm key, e.g. "counts".
    :param value: its value as stored, e.g. "log1p" or False.
    :return: e.g. "log scale".
    """
    return ARM_VALUES.get((setting, str(value)), str(value))


def round_label(rnd: str, prompt_version: str, variants_in_round: int) -> str:
    """Plain label for one prompt round.

    Round 1 ran two prompt versions; the CSV marks them "(first 8)" and
    "(all 22)" after the hash.

    :param rnd: round id, e.g. "r1".
    :param prompt_version: the CSV prompt_version, e.g. "64c7c902e4f2 (first 8)".
    :param variants_in_round: how many rows the round has.
    :return: e.g. "Round 1 (8 examples)", "Round 1 (all 22 examples)" or "Round 2".
    """
    base = f"Round {rnd.removeprefix('r')}"
    if variants_in_round < 2:
        return base
    match = re.search(r"\((first|all) (\d+)\)", prompt_version)
    if not match:
        return base
    kind, count = match.groups()
    return f"{base} ({'all ' if kind == 'all' else ''}{count} examples)"


def kind_name(kind: str) -> str:
    """How a judged item's kind is shown.

    :param kind: e.g. "control_premise".
    :return: e.g. "Planted: dropped set-up".
    """
    return KINDS.get(kind, kind)


def check_names(failed: str) -> str:
    """Failed judge checks, comma-joined, in plain words.

    :param failed: e.g. "rules_ok" or "faithful, rules_ok" or "".
    :return: e.g. "style rules" or "accuracy, style rules" or "".
    """
    return ", ".join(CHECKS.get(c.strip(), c.strip()) for c in failed.split(",") if c.strip())


def mapped(values, mapping: Mapping[str, str]) -> list[str]:
    """Map each value through a dict, keeping unmapped values as they are.

    :param values: any iterable of keys.
    :param mapping: key -> display name.
    :return: the display names.
    """
    return [mapping.get(v, v) for v in values]
