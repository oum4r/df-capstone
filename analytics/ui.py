"""Page helpers: definitions, notes and static tables.

Tables are printed as static HTML (styled by theme.CSS) so long text wraps
instead of scrolling sideways.
"""

import html
import math
from collections.abc import Callable, Mapping, Sequence

import pandas as pd
import streamlit as st

import content

NA = "n/a"
DEFINITIONS_LABEL = "What these measures mean"
NOTES_LABEL = "Notes on the data"
Formatter = str | Callable[[object], str]


def is_missing(value: object) -> bool:
    """Whether a cell value is None or a float NaN.

    :param value: any cell value.
    :return: True for None or NaN.
    """
    return value is None or (isinstance(value, float) and math.isnan(value))


def format_cell(value: object, fmt: Formatter | None = None) -> str:
    """One cell as text: "n/a" when missing, else fmt applied, else str().

    :param value: the cell value.
    :param fmt: a str.format pattern such as "{:.4f}", or a callable.
    :return: the display text, not yet HTML-escaped.
    """
    if is_missing(value):
        return NA
    if fmt is None:
        return str(value)
    return fmt(value) if callable(fmt) else fmt.format(value)


def html_table(
    df: pd.DataFrame,
    labels: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    formats: Mapping[str, Formatter] | None = None,
    numeric: Sequence[str] = (),
    wrap_anywhere: Sequence[str] = (),
    highlight: Sequence[bool] | None = None,
) -> str:
    """A static, wrapping HTML table in the Folio style.

    :param df: the rows to print.
    :param labels: column -> header text; defaults to the column name.
    :param order: columns to print, in order; defaults to all.
    :param formats: column -> format pattern or callable.
    :param numeric: columns set right-aligned with tabular figures.
    :param wrap_anywhere: columns whose long unbroken strings (URLs) may break at any character.
    :param highlight: one flag per row; True rows take the accent (the app's model).
    :return: the table as an HTML string, every cell escaped.
    """
    labels, formats = labels or {}, formats or {}
    columns = list(order) if order else list(df.columns)

    def cls(col: str) -> str:
        names = (["num"] if col in numeric else []) + (["wrap"] if col in wrap_anywhere else [])
        return f' class="{" ".join(names)}"' if names else ""

    head = "".join(f"<th{cls(c)}>{html.escape(labels.get(c, c))}</th>" for c in columns)
    body = []
    for i, (_, row) in enumerate(df.iterrows()):
        mark = ' class="shipped"' if highlight is not None and highlight[i] else ""
        cells = "".join(f"<td{cls(c)}>{html.escape(format_cell(row[c], formats.get(c)))}</td>" for c in columns)
        body.append(f"<tr{mark}>{cells}</tr>")
    table = f'<table class="folio-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    # A wide table scrolls inside its own box rather than spilling past the text column.
    return f'<div class="folio-scroll">{table}</div>'


def static_table(df: pd.DataFrame, **kwargs) -> None:
    """Print html_table output on the page.

    :param df: the rows to print.
    :param kwargs: passed to html_table.
    """
    st.html(html_table(df, **kwargs))


def split_definition(text: str) -> tuple[str, str]:
    """Split "MAE, mean absolute error: on average, ..." into measure and meaning.

    :param text: a content.METRICS text, "measure: meaning".
    :return: (measure, meaning with its first letter capitalised).
    """
    measure, _, meaning = text.partition(": ")
    return measure, meaning[:1].upper() + meaning[1:]


def explain(*keys: str) -> None:
    """Define a section's measures in a closed expander: a sentence for one, a table for several.

    :param keys: keys of content.METRICS.
    """
    with st.expander(DEFINITIONS_LABEL, expanded=False):
        if len(keys) == 1:
            st.markdown(content.METRICS[keys[0]])
        else:
            rows = [split_definition(content.METRICS[k]) for k in keys]
            static_table(
                pd.DataFrame(rows, columns=["measure", "meaning"]),
                labels={"measure": "Measure", "meaning": "Plain meaning"},
            )


class Notes:
    """Method notes gathered through a page and printed in one closed expander at its foot."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, section: str, text: str) -> None:
        """Queue one note.

        :param section: the section it belongs to, e.g. "Learning curve".
        :param text: the note.
        """
        self.items.append((section, text))

    def render(self) -> None:
        """Print the queued notes, if any."""
        if not self.items:
            return
        with st.expander(NOTES_LABEL, expanded=False):
            for section, text in self.items:
                st.markdown(f"**{section}.** {text}")


def dollars(value: float) -> str:
    """US dollars to 6 decimals, e.g. $0.000984.

    :param value: an amount in dollars.
    :return: the formatted amount.
    """
    return f"${value:.6f}"
