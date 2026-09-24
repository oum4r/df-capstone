"""Folio theme tokens, the page CSS and the Altair chart theme.

Folio: an off-white ground, ink text and rules, one red accent and five
dark ramp fills. The accent marks the app's model and nothing else, so every
other series takes a ramp fill or ink. Text stays at 15 px or more, in full
ink with no opacity (the 18 px root is set in .streamlit/config.toml).
"""

import altair as alt
import streamlit as st

GROUND = "#F5F1E6"
PAPER_SECONDARY = "#EAE3D2"
INK = "#1B1815"
ACCENT = "#A3202B"
ACCENT_TINT = "#E9D2CA"
LINK = "#2A4B7C"
GILT = "#EFD9A0"
RAMP = ["#7A2230", "#6B3D1E", "#585020", "#333A4C", "#100F0D"]
NEUTRAL_FILL = RAMP[3]  # the teal-grey step: the default bar colour
# Series colours after the accent, in order; the red-brown ramp step comes
# last so it never sits next to the accent unless it has to.
SERIES = [RAMP[3], RAMP[1], RAMP[4], RAMP[2], RAMP[0]]
FONT = "EB Garamond"
RADIUS_PX = 2

# Chart type sizes in px (Vega draws in px, not rem).
CHART_LABEL_PX = 15
CHART_TITLE_PX = 16

# Small caps on labels and section heads, tabular figures on numbers, 1 px
# ink rules, no shadows. The header with
# the top navigation is 3.75rem tall, so 4.5rem of top padding leaves about
# 0.75rem of space under it.
CSS = f"""
<style>
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 1400px !important; margin: 0 auto;
  padding: 4.5rem 2rem 1.5rem !important;
}}
p, li {{ line-height: 1.5; }}
[data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] :is(strong, b) {{ font-weight: 700; }}
h1 {{ padding-bottom: 0.25rem !important; }}
h2, [data-testid="stHeading"] h2 {{
  font-variant-caps: small-caps; letter-spacing: 0.02em;
  border-bottom: 1px solid {INK}; padding-bottom: 0.25rem !important;
  margin-top: 0.75rem !important; margin-bottom: 0.5rem !important;
}}
[data-baseweb="tab"], [data-testid="stSegmentedControl"] button,
[data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p,
[data-testid="stExpander"] summary p,
[data-testid="stNavSectionHeader"], [data-testid="stTopNavLink"] span {{
  font-variant-caps: small-caps; letter-spacing: 0.02em;
}}
/* Full ink everywhere: Streamlit fades captions, labels and slider ticks. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] *,
[data-testid="stSliderTickBar"], [data-testid="stSliderTickBar"] *,
[data-testid="stSliderThumbValue"], [data-testid="stExpander"] summary *,
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] *,
[data-testid="stTopNavLink"], [data-testid="stTopNavLink"] * {{
  color: {INK} !important; opacity: 1 !important;
}}
[data-testid="stCaptionContainer"] {{ font-size: 0.875rem !important; margin-bottom: 0.5rem; }}
[data-testid="stWidgetLabel"] p {{ font-size: 0.95rem !important; font-weight: 700; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.9rem !important; }}
[data-testid="stMetricDelta"] {{ font-size: 0.9rem !important; background: none; padding: 0; border-radius: 0; }}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {{ white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
[data-testid="stSliderTickBar"], [data-testid="stSliderThumbValue"],
[data-testid="stTooltipContent"] {{ font-size: 0.875rem !important; }}
[data-testid="stTopNavLink"], [data-testid="stTopNavLink"] span {{ font-size: 1rem !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ background: {ACCENT_TINT}; }}
[data-testid="stTopNavLink"][aria-current="page"] * {{ color: {ACCENT} !important; font-weight: 700; }}
::placeholder {{ color: {INK} !important; opacity: 1 !important; }}
/* Streamlit's smallest size (0.75rem, 13.5 px at this root) is raised to the 15 px floor. */
[data-testid="stMain"] small, [data-baseweb="menu"] li,
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{
  font-size: 0.875rem !important; color: {INK} !important; opacity: 1 !important;
}}
/* Chosen multiselect tags: accent text on the tint, the Folio chosen-state pair (5.19:1). */
[data-testid="stMultiSelectTagsContainer"] [role="group"] > span {{
  background: {GROUND} !important; color: {INK} !important; border: 1px solid {INK}; border-radius: {RADIUS_PX}px;
}}
[data-testid="stMultiSelectTagsContainer"] [role="group"] > span * {{ color: {INK} !important; fill: {INK} !important; opacity: 1 !important; }}
[data-testid="stMetricValue"], [data-testid="stMetricDelta"], .tabular {{ font-variant-numeric: tabular-nums; }}
[data-testid="stMetric"] {{ border: 1px solid {INK}; border-radius: {RADIUS_PX}px; padding: 0.9rem 1.1rem; }}
[data-testid="stExpander"] details {{ border-radius: {RADIUS_PX}px; }}
[data-testid="stExpander"] summary p {{ font-weight: 700; font-size: 1rem; }}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stHeader"],
[data-baseweb="popover"] > div {{ box-shadow: none !important; }}
:focus-visible {{ outline: 2px solid {LINK}; outline-offset: 2px; }}
/* Chart tooltips (vega-tooltip): ink on the ground, no grey keys. */
#vg-tooltip-element, .vg-tooltip {{
  font-family: "{FONT}", serif !important; font-size: 15px !important; color: {INK} !important;
  background: {GROUND} !important; border: 1px solid {INK} !important; box-shadow: none !important;
}}
#vg-tooltip-element td, .vg-tooltip td {{ color: {INK} !important; font-size: 15px !important; }}
#vg-tooltip-element td.key, .vg-tooltip td.key {{ font-weight: 700; }}
.folio-scroll {{ overflow-x: auto; max-width: 100%; }}
.folio-table {{
  width: 100%; border-collapse: collapse; margin: 0.25rem 0 0.25rem;
  border-top: 1px solid {INK}; border-bottom: 1px solid {INK};
  font-size: 0.95rem; font-weight: 500; line-height: 1.4; color: {INK};
}}
.folio-table th {{
  background: {PAPER_SECONDARY}; color: {INK}; font-weight: 700;
  font-variant-caps: small-caps; letter-spacing: 0.02em; text-align: left; vertical-align: bottom;
  border-bottom: 1px solid {INK}; padding: 0.45rem 0.8rem 0.35rem 0.5rem;
}}
.folio-table td {{
  vertical-align: top; border-top: 1px solid {INK}; padding: 0.4rem 0.8rem 0.4rem 0.5rem;
  overflow-wrap: break-word;
}}
.folio-table .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.folio-table th.num {{ white-space: normal; }}
.folio-table .wrap {{ overflow-wrap: anywhere; }}
.folio-table tr.shipped td {{ background: {ACCENT_TINT}; color: {ACCENT}; font-weight: 700; }}
</style>
"""


def inject_css() -> None:
    """Write the Folio CSS block into the page once per run."""
    st.html(CSS)


@alt.theme.register("folio", enable=True)
def folio_chart_theme() -> alt.theme.ThemeConfig:
    """Altair theme: EB Garamond, a transparent ground, ink axes and labels
    (15 px at 500, titles 16 px at 700), secondary-paper gridlines and the
    five ramp fills as the category range.

    :return: an Altair theme config dict.
    """
    axis = {
        "domainColor": INK,
        "tickColor": INK,
        "labelColor": INK,
        "titleColor": INK,
        "gridColor": PAPER_SECONDARY,
        "labelFont": FONT,
        "titleFont": FONT,
        "labelFontSize": CHART_LABEL_PX,
        "labelFontWeight": 500,
        "titleFontSize": CHART_TITLE_PX,
        "titleFontWeight": 700,
        # End labels sit inside the axis and overlapping labels drop out, so
        # neighbours such as "0.00" and "0.01" never print as one string.
        "labelOverlap": "greedy",
        "labelFlush": True,
        "labelPadding": 4,
        "labelLimit": 360,
    }
    quantitative = {"tickCount": 6, "labelOverlap": "greedy", "labelFlush": True}
    return {
        "config": {
            "background": "transparent",
            "font": FONT,
            "view": {"stroke": None},
            "axis": axis,
            "axisQuantitative": quantitative,
            "legend": {
                "labelFont": FONT,
                "titleFont": FONT,
                "labelColor": INK,
                "titleColor": INK,
                "labelFontSize": CHART_LABEL_PX,
                "labelFontWeight": 500,
                "titleFontSize": CHART_TITLE_PX,
                "titleFontWeight": 700,
                "labelLimit": 360,
            },
            "title": {"font": FONT, "color": INK, "fontSize": 17, "fontWeight": 700, "anchor": "start"},
            "range": {"category": RAMP},
            "bar": {"cornerRadius": 0},
            "line": {"strokeWidth": 2},
            "point": {"size": 60, "filled": True},
        }
    }


def series_colours(names: list[str], shipped: str | None) -> list[str]:
    """One colour per series: the accent for the shipped model, ramp fills for the rest.

    :param names: series names in legend order.
    :param shipped: the name that takes the accent, or None.
    :return: colours, same order as names.
    """
    others = iter(SERIES * (len(names) // len(SERIES) + 1))
    return [ACCENT if n == shipped else next(others) for n in names]


def show_chart(chart: alt.Chart) -> None:
    """Render an Altair chart with the Folio theme rather than Streamlit's own.

    :param chart: any Altair chart.
    """
    st.altair_chart(chart, theme=None, width="stretch")
