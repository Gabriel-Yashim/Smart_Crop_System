"""Shared paths, theming and small HTML components for the Agriculture Smart System app.

Colors come from themes.py. Change the theme with:  python apply_theme.py <theme name>
"""

import base64
import html
import io
import json
from pathlib import Path

import streamlit as st

import themes

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"          # crop recommendation files (scaler, encoder, model, features)
DISEASE_DIR = ROOT / "models" / "disease"   # model.onnx, class_names.json, preprocess.json, metrics.json, reports/

APP_NAME = "Agriculture Smart System"
# Edit this line to change the credit shown at the bottom of every page.
FOOTER_TEXT = ("Crop Prediction App by OLADIPE EBENEZER OLUWOLE  |  Mat. No. PHD23CSC009  ·  Built with Streamlit  ·  "
               "Crop model trained on the African Crop Recommendation Dataset")

# True shows a theme picker at the top of the sidebar, for trying themes live. Set it to False once you
# have chosen a theme, so visitors do not see it.
THEME_PICKER = True

# Filled in by app.py so pages can link to each other.
PAGES = {}

# Placeholders written as @name@ are replaced with the active theme's colors.
CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=DM+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], .stMarkdown, button, input, textarea {
    font-family: 'DM Sans', system-ui, sans-serif;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: @bg@ !important; color: @text@; }
[data-testid="stHeader"] { background: @bg@ !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child { background: @sidebar@ !important; }
.stApp p, .stApp li, .stApp label, .stApp span, .stApp summary, [data-testid="stWidgetLabel"] p { color: @text@; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: @ink@; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4 { font-family: 'Fraunces', Georgia, serif; font-weight: 600; letter-spacing: -0.01em; }

/* native widgets, so they follow the theme even when config.toml uses another one */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
    background: @surface@ !important; border-color: @border@ !important; color: @ink@ !important; }
[data-baseweb="input"] input, [data-baseweb="base-input"] input, [data-baseweb="select"] input, textarea {
    color: @ink@ !important; -webkit-text-fill-color: @ink@ !important; background: transparent !important; }
[data-baseweb="select"] span, [data-baseweb="select"] div { color: @ink@; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] { background: @surface@ !important; color: @ink@ !important; }
[data-testid="stNumberInput"] button { background: @track@ !important; color: @ink@ !important; border-color: @border@ !important; }
[data-testid="stFileUploaderDropzone"] { background: @surface@ !important; border: 1px dashed @rule@ !important; border-radius: 14px; }
[data-testid="stFileUploaderDropzone"] * { color: @text@ !important; }
[data-testid="stFileUploaderDropzone"] button { background: @track@ !important; color: @ink@ !important; border: 1px solid @border@ !important; }
[data-testid="stCameraInput"] > div, [data-testid="stCameraInput"] button { background: @surface@; color: @ink@; border-color: @border@; }
[data-baseweb="tab-list"] { gap: 0.4rem; }
button[data-baseweb="tab"] { background: transparent !important; }
button[data-baseweb="tab"] p { color: @muted@ !important; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] p { color: @accent@ !important; }
[data-baseweb="tab-highlight"] { background: @accent@ !important; }
[data-baseweb="tab-border"] { background: @border@ !important; }
[data-testid="stExpander"] { background: @surface@ !important; border: 1px solid @border@ !important; border-radius: 14px; }
[data-testid="stExpander"] details summary p, [data-testid="stExpander"] details summary span { color: @ink@ !important; }
[data-testid="stForm"] { background: @surface@ !important; border: 1px solid @border@ !important; border-radius: 16px; padding: 1.2rem 1.4rem; width: 100%; }
[data-testid="stAlert"] { background: @soft@; border: 1px solid @border@; border-radius: 12px; }
[data-testid="stPageLink"] a { background: @soft@; border: 1px solid @border@; border-radius: 999px; padding: 0.4rem 1rem; text-decoration: none; }
[data-testid="stPageLink"] a p { color: @accent@ !important; font-weight: 600; }
[data-testid="stFormSubmitButton"] button, .stButton button, button[kind="primary"] {
    background: @accent@ !important; border: 1px solid @accent@ !important; color: @accent_ink@ !important;
    border-radius: 999px; font-weight: 600; padding: 0.55rem 1.4rem; }
[data-testid="stFormSubmitButton"] button p, .stButton button p { color: @accent_ink@ !important; }
[data-testid="stFormSubmitButton"] button { width: 100%; }
code { color: @accent@; background: @soft@; }

.as-eyebrow { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: @accent@; margin-bottom: 0.35rem; }
.as-title { font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 2.35rem; line-height: 1.1; color: @ink@; margin: 0 0 0.5rem 0; padding: 0; }
.as-hero-title { font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 3.4rem; line-height: 1.04; letter-spacing: -0.02em; color: @ink@; margin: 0 0 0.9rem 0; padding: 0; }
.as-lede { font-size: 1.08rem; line-height: 1.55; color: @text@; max-width: 48rem; margin: 0 0 1.2rem 0; }
.as-rule { height: 1px; background: @rule@; margin: 0.4rem 0 1.4rem 0; }
.as-section { font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 1.45rem; color: @ink@; margin: 1.8rem 0 0.2rem 0; }
.as-section-sub { color: @muted@; font-size: 0.95rem; margin: 0 0 0.9rem 0; }
.as-muted { color: @muted@ !important; font-size: 0.92rem; line-height: 1.5; }
.as-small { color: @muted@ !important; font-size: 0.82rem; line-height: 1.45; margin-top: 0.8rem; }

.as-card { background: @surface@; border: 1px solid @border@; border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.as-card h3 { margin: 0 0 0.4rem 0; font-size: 1.4rem; color: @ink@; }
.as-card p { margin: 0 0 0.7rem 0; color: @text@; line-height: 1.55; }
.as-card p:last-child { margin-bottom: 0; }
.as-card b { color: @ink@; }
.as-card-label { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: @muted@; margin-bottom: 0.5rem; }
.as-card img { width: 100%; height: auto; display: block; }

.as-hero-card { display: flex; align-items: center; justify-content: space-between; gap: 1rem; background: @surface@; border: 1px solid @border@; border-left: 6px solid @accent@; border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; }
.as-result-name { font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 2.5rem; line-height: 1.08; color: @ink@; margin: 0.1rem 0 0.7rem 0; }

.as-ring { width: 104px; height: 104px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.as-ring-in { width: 78px; height: 78px; border-radius: 50%; background: @surface@; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.as-ring-in b { font-family: 'Fraunces', Georgia, serif; font-size: 1.35rem; color: @ink@; line-height: 1; }
.as-ring-in span { font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase; color: @muted@; margin-top: 2px; }

.as-pill { display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin: 0 0.4rem 0.3rem 0; }
.as-pill.ok { background: @ok_bg@; color: @ok_fg@; }
.as-pill.warn { background: @warn_bg@; color: @warn_fg@; }
.as-pill.bad { background: @bad_bg@; color: @bad_fg@; }
.as-pill.neutral { background: @neutral_bg@; color: @neutral_fg@; }

.as-chip { display: inline-block; padding: 0.18rem 0.65rem; margin: 0 0.35rem 0.35rem 0; border: 1px solid @border@; border-radius: 8px; font-size: 0.8rem; color: @text@; background: @chip_bg@; }
.as-chip b { color: @ink@; font-weight: 600; margin-right: 0.25rem; }
.as-chip.ok { background: @ok_bg@; color: @ok_fg@; border-color: @ok_bg@; }

.as-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.9rem; margin: 0.4rem 0 1rem 0; }
.as-kpi { background: @surface@; border: 1px solid @border@; border-radius: 14px; padding: 0.95rem 1.1rem; }
.as-kpi-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: @muted@; }
.as-kpi-value { font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 1.9rem; line-height: 1.15; color: @ink@; margin-top: 0.15rem; }
.as-kpi-note { font-size: 0.8rem; color: @muted@; margin-top: 0.1rem; }
.as-kpis.compact { grid-template-columns: 1fr 1fr; gap: 0.6rem; }
.as-kpis.compact .as-kpi { padding: 0.65rem 0.8rem; border-radius: 12px; }
.as-kpis.compact .as-kpi-value { font-size: 1.3rem; }

.as-bars { margin-top: 0.3rem; }
.as-bar-row { display: grid; grid-template-columns: minmax(130px, 40%) 1fr 54px; gap: 0.75rem; align-items: center; margin: 0.5rem 0; }
.as-bar-label { font-size: 0.92rem; color: @ink@; line-height: 1.25; overflow-wrap: anywhere; }
.as-track { height: 10px; background: @track@; border-radius: 999px; overflow: hidden; }
.as-fill { height: 100%; background: @accent@; border-radius: 999px; }
.as-fill.alt { background: @alt@; }
.as-bar-pct { font-size: 0.85rem; color: @text@; text-align: right; font-variant-numeric: tabular-nums; }

.as-kv { display: grid; grid-template-columns: 1.3fr 1fr auto; gap: 0.6rem; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid @row_line@; font-size: 0.92rem; }
.as-kv:last-child { border-bottom: none; }
.as-kv-k { color: @text@; }
.as-kv-v { color: @ink@; font-weight: 600; }
.as-kv-p { text-align: right; }

.as-steps { counter-reset: step; margin: 0.2rem 0 0 0; padding: 0; list-style: none; }
.as-steps li { position: relative; padding: 0 0 0.9rem 2.5rem; color: @text@; line-height: 1.5; font-size: 0.95rem; }
.as-steps li b { color: @ink@; }
.as-steps li::before { counter-increment: step; content: counter(step); position: absolute; left: 0; top: -0.1rem; width: 1.75rem; height: 1.75rem; border-radius: 50%; background: @soft@; color: @accent@; font-weight: 600; font-size: 0.85rem; display: flex; align-items: center; justify-content: center; }

.as-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.as-table th { text-align: left; color: @muted@; font-weight: 600; font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; padding: 0.4rem 0.3rem; border-bottom: 1px solid @rule@; }
.as-table td { padding: 0.5rem 0.3rem; border-bottom: 1px solid @row_line@; color: @ink@; }
.as-table td.r { color: @muted@; }

.as-side-brand { font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 1.25rem; color: @ink@; line-height: 1.15; margin-bottom: 0.1rem; }
.as-side-title { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: @accent@; margin: 1.3rem 0 0.5rem 0; }

.as-footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid @rule@; color: @muted@; font-size: 0.8rem; }

@media (max-width: 640px) {
  .as-hero-title { font-size: 2.3rem; }
  .as-title { font-size: 1.9rem; }
  .as-result-name { font-size: 1.9rem; }
  .as-bar-row { grid-template-columns: minmax(100px, 38%) 1fr 48px; gap: 0.5rem; }
  .as-ring { width: 84px; height: 84px; }
  .as-ring-in { width: 62px; height: 62px; }
  .as-kv { grid-template-columns: 1fr 1fr; }
  .as-kv-p { grid-column: 1 / -1; text-align: left; }
}
</style>
"""


# ==== theme
def active_theme_name():
    name = st.session_state.get("theme_name", themes.DEFAULT_THEME)
    return name if name in themes.THEMES else themes.DEFAULT_THEME


def palette():
    """Colors of the active theme."""
    return themes.tokens(active_theme_name())


def build_css(name):
    css = CSS_TEMPLATE
    for key, value in themes.tokens(name).items():
        css = css.replace(f"@{key}@", str(value))
    return css


def init_theme():
    """Call once at the top of app.py, before the pages run."""
    names = list(themes.THEMES)
    if "theme_name" not in st.session_state or st.session_state["theme_name"] not in themes.THEMES:
        st.session_state["theme_name"] = themes.DEFAULT_THEME
    if THEME_PICKER:
        st.sidebar.selectbox(
            "Theme", names, key="theme_name",
            format_func=lambda n: f"{themes.THEMES[n]['label']} ({themes.THEMES[n]['mode']})",
            help="Try the colors live. To keep one for good, run: python apply_theme.py <name>",
        )
    st.markdown(build_css(active_theme_name()), unsafe_allow_html=True)


# ==== helpers
def esc(value):
    return html.escape(str(value))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def page_header(eyebrow, title, subtitle=""):
    parts = [f'<div class="as-eyebrow" style="margin-top: 1rem">{esc(eyebrow)}</div>', f'<h1 class="as-title">{esc(title)}</h1>']
    if subtitle:
        parts.append(f'<p class="as-lede">{esc(subtitle)}</p>')
    parts.append('<div class="as-rule"></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title, subtitle=""):
    html_ = f'<div class="as-section">{esc(title)}</div>'
    if subtitle:
        html_ += f'<div class="as-section-sub">{esc(subtitle)}</div>'
    st.markdown(html_, unsafe_allow_html=True)


def pill(text, tone="neutral"):
    return f'<span class="as-pill {tone}">{esc(text)}</span>'


def chips(pairs):
    """pairs: list of (name, value). Returns HTML."""
    return "".join(f'<span class="as-chip"><b>{esc(k)}</b>{esc(v)}</span>' for k, v in pairs)

def show_fig(fig):
    import matplotlib.pyplot as plt

    st.pyplot(fig)
    plt.close(fig)

def plain_chips(names, tone=""):
    cls = f"as-chip {tone}".strip()
    return "".join(f'<span class="{cls}">{esc(n)}</span>' for n in names)


def kpis(items, compact=False):
    """items: list of (label, value, note). Returns HTML."""
    cells = "".join(
        f'<div class="as-kpi"><div class="as-kpi-label">{esc(label)}</div>'
        f'<div class="as-kpi-value">{esc(value)}</div>'
        + (f'<div class="as-kpi-note">{esc(note)}</div>' if note else "")
        + "</div>"
        for label, value, note in items
    )
    return f'<div class="as-kpis{" compact" if compact else ""}">{cells}</div>'


def bars(items):
    """items: list of (label, probability between 0 and 1). The first row is highlighted."""
    rows = []
    for i, (label, p) in enumerate(items):
        pct = max(0.0, min(100.0, float(p) * 100))
        fill = "as-fill" if i == 0 else "as-fill alt"
        rows.append(
            f'<div class="as-bar-row"><div class="as-bar-label">{esc(label)}</div>'
            f'<div class="as-track"><div class="{fill}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="as-bar-pct">{pct:.1f}%</div></div>'
        )
    return f'<div class="as-bars">{"".join(rows)}</div>'


def kv_rows(rows):
    """rows: list of (label, value, optional pill html)."""
    return "".join(
        f'<div class="as-kv"><div class="as-kv-k">{esc(k)}</div><div class="as-kv-v">{esc(v)}</div>'
        f'<div class="as-kv-p">{pillhtml}</div></div>'
        for k, v, pillhtml in rows
    )


def steps(items):
    """items: list of (bold title, text)."""
    lis = "".join(f"<li><b>{esc(t)}</b> {esc(text)}</li>" for t, text in items)
    return f'<ol class="as-steps">{lis}</ol>'


def table_html(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="as-table"><tr>{head}</tr>{body}</table>'


def ring_color(pct):
    p = palette()
    return p["good"] if pct >= 80 else p["mid"] if pct >= 50 else p["bad"]


def hero_card(label, name, pills_html, pct, ring_label="confidence"):
    color = ring_color(pct)
    track = palette()["track"]
    p = max(0, min(100, pct))
    ring = (f'<div class="as-ring" style="background:conic-gradient({color} 0 {p:.1f}%, {track} {p:.1f}% 100%)">'
            f'<div class="as-ring-in"><b>{p:.0f}%</b><span>{esc(ring_label)}</span></div></div>')
    st.markdown(
        f'<div class="as-hero-card"><div><div class="as-card-label">{esc(label)}</div>'
        f'<div class="as-result-name">{esc(name)}</div>{pills_html}</div>{ring}</div>',
        unsafe_allow_html=True,
    )


def card(inner_html):
    st.markdown(f'<div class="as-card">{inner_html}</div>', unsafe_allow_html=True)


def fig_html(fig, alt="chart"):
    """Matplotlib figure as an inline image, so a chart can sit inside a themed HTML card."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{data}">'


def chart_card(label, fig, alt="chart"):
    card(f'<div class="as-card-label">{esc(label)}</div>' + fig_html(fig, alt))


def sidebar_brand(subtitle=""):
    st.sidebar.markdown(
        f'<div class="as-side-brand">{esc(APP_NAME)}</div>'
        + (f'<div class="as-muted">{esc(subtitle)}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def sidebar_title(text):
    st.sidebar.markdown(f'<div class="as-side-title">{esc(text)}</div>', unsafe_allow_html=True)


def footer():
    st.markdown(f'<div class="as-footer">{esc(FOOTER_TEXT)}</div>', unsafe_allow_html=True)
