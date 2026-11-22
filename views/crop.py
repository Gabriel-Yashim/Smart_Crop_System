"""Crop recommendation page (machine learning on soil and climate data)."""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import charts
import knowledge as kb
import ui


@st.cache_resource(show_spinner=False)
def load_artifacts():
    base = ui.ARTIFACTS_DIR
    model = joblib.load(base / "crop_model_Stacking Classifier.pkl")
    scaler = joblib.load(base / "scaler.pkl")
    le = joblib.load(base / "label_encoder.pkl")
    features = joblib.load(base / "features.pkl")
    return model, scaler, le, features


def predict(model, scaler, le, values):
    """values: [N, P, K, temperature, humidity, ph, rainfall]. Same logic as the original app."""
    raw = np.array([values])
    scaled = scaler.transform(raw)
    pred_label = model.predict(scaled)[0]
    pred_name = le.inverse_transform([pred_label])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(scaled)[0]
        proba = np.exp(scores) / np.exp(scores).sum()
    else:
        proba = np.zeros(len(le.classes_))
        proba[pred_label] = 1.0

    order = np.argsort(proba)[::-1]
    ranked = [(str(le.classes_[i]), float(proba[i])) for i in order]
    return {"name": str(pred_name), "confidence": ranked[0][1], "ranked": ranked}


def _fmt(v):
    return f"{v:g}"


def snapshot_rows(values):
    """Plain language reading of each input, relative to the typical range in the training data."""
    spec = {s[0]: s for s in kb.INPUT_SPECS}
    rows = []
    for key in ("N", "P", "K"):
        _, label, lo, hi, unit = spec[key]
        rows.append((label, f"{_fmt(values[key])} {unit}", ui.pill(kb.nutrient_level(values[key], lo, hi), "neutral")))
    rows.append(("Soil pH", _fmt(values["ph"]), ui.pill(kb.ph_category(values["ph"]), "neutral")))
    rows.append(("Temperature", f"{_fmt(values['temperature'])} °C", ui.pill(kb.temperature_category(values["temperature"]), "neutral")))
    rows.append(("Humidity", f"{_fmt(values['humidity'])} %", ui.pill(kb.humidity_category(values["humidity"]), "neutral")))
    rows.append(("Rainfall", f"{_fmt(values['rainfall'])} mm", ui.pill(kb.rainfall_category(values["rainfall"]), "neutral")))
    return rows


def range_rows(values):
    rows = []
    for key, label, lo, hi, unit in kb.INPUT_SPECS:
        v = values[key]
        suffix = f" {unit}" if unit else ""
        rows.append((label, v, lo, hi, f"{_fmt(v)}{suffix}", f"{_fmt(lo)}{suffix}", f"{_fmt(hi)}{suffix}"))
    return rows


def confidence_reading(res):
    ranked = res["ranked"]
    name, conf = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    close = [c for c, p in ranked[1:5] if p >= conf - 0.15]
    if conf >= 0.8:
        text = f"The model is strongly in favor of {name.title()}."
    elif conf >= 0.5:
        text = f"The model leans toward {name.title()}, but it is not a landslide."
    else:
        text = f"No crop stands far ahead. {name.title()} is only the best of several similar fits."
    if runner:
        text += f" It leads the runner-up, {runner[0].title()}, by {(conf - runner[1]) * 100:.1f} percentage points."
    return text, close


def render_results(saved):
    res, values = saved["res"], saved["values"]
    ranked = res["ranked"]
    name, conf = res["name"], res["confidence"]
    tone = "ok" if conf >= 0.8 else "warn" if conf >= 0.5 else "bad"

    ui.section("Your result")
    a, b = st.columns([2, 3], gap="large")
    with a:
        label = "High confidence" if conf >= 0.8 else "Moderate confidence" if conf >= 0.5 else "Low confidence"
        ui.hero_card("Recommended crop", name.title(), ui.pill(label, tone), conf * 100)
        reading, close = confidence_reading(res)
        extra = ""
        if close:
            extra = '<p class="as-muted">Also worth considering:</p>' + ui.plain_chips([c.title() for c in close])
        ui.card(f'<div class="as-card-label">How sure is the model</div><p>{ui.esc(reading)}</p>{extra}')
    with b:
        with st.container(border=True):
            st.markdown('<div class="as-card-label">Top 5 options</div>', unsafe_allow_html=True)
            ui.show_fig(charts.hbar([(c.title(), p) for c, p in ranked[:5]]))

    ui.section("Your conditions", "How your inputs compare with the range the model was trained on.")
    c, d = st.columns([3, 2], gap="large")
    with c:
        with st.container(border=True):
            st.markdown('<div class="as-card-label">Inputs against typical ranges</div>', unsafe_allow_html=True)
            ui.show_fig(charts.range_profile(range_rows(values)))
    with d:
        ui.card('<div class="as-card-label">Soil and climate snapshot</div>' + ui.kv_rows(snapshot_rows(values))
                + '<p class="as-small">Nutrient levels are relative to the typical range in the training data. '
                  "The other labels are indicative.</p>")

    ui.section(f"Growing {name.title()}")
    e, f = st.columns(2, gap="large")
    tip = kb.CROP_TIPS.get(
        name.lower(),
        f"{name.title()} grows best when soil nutrients, pH and rainfall stay close to the values you entered.",
    )
    with e:
        ui.card(f'<div class="as-card-label">Growing notes</div><p>{ui.esc(tip)}</p>')
    with f:
        ui.card('<div class="as-card-label">Before you decide</div>'
                "<p>Treat this as a starting point. Check seed availability, market demand and your local "
                "planting calendar, and confirm soil values with a lab test when you can.</p>")

    with st.expander("All crop probabilities"):
        ui.show_fig(charts.hbar([(c.title(), p) for c, p in ranked]))


def render_welcome(le):
    crops = [str(c).title() for c in le.classes_]
    ui.section("How it works")
    a, b = st.columns([3, 2], gap="large")
    with a:
        ui.card(
            '<div class="as-card-label">Three steps</div>'
            + ui.steps([
                ("Enter your conditions.", "Soil nitrogen, phosphorus, potassium and pH, plus temperature, humidity and rainfall."),
                ("Get a recommendation.", "The model scores every crop it knows and ranks them from best to weakest fit."),
                ("Check the details.", "See the top 5, how your inputs compare with typical ranges, and growing notes."),
            ])
        )
    with b:
        ui.card(f'<div class="as-card-label">{len(crops)} crops it can recommend</div>' + ui.plain_chips(sorted(crops)))

    ui.section("Typical ranges", "The ranges the model has seen. Values far outside them make predictions less reliable.")
    table = pd.DataFrame(
        [{"Input": label, "Typical low": lo, "Typical high": hi, "Unit": unit or "pH units"}
         for _k, label, lo, hi, unit in kb.INPUT_SPECS]
    )
    st.dataframe(table, hide_index=True)


def render_sidebar(model, le, features, saved):
    ui.sidebar_brand("Crop recommendation")

    ui.sidebar_title("Your inputs")
    if saved:
        rows = "".join(
            f"<tr><td>{ui.esc(label)}</td><td><b>{ui.esc(_fmt(saved['values'][key]))}</b> {ui.esc(unit)}</td>"
            f'<td class="r">{ui.esc(_fmt(lo))} to {ui.esc(_fmt(hi))}</td></tr>'
            for key, label, lo, hi, unit in kb.INPUT_SPECS
        )
        st.sidebar.markdown(
            f'<table class="as-table"><tr><th>Input</th><th>Value</th><th>Typical</th></tr>{rows}</table>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown('<p class="as-muted">Submit the form to see your inputs here.</p>', unsafe_allow_html=True)

    ui.sidebar_title("Latest result")
    if saved:
        res = saved["res"]
        st.sidebar.markdown(
            f'<div class="as-card" style="margin-bottom:0.5rem"><div class="as-card-label">Recommended</div>'
            f'<div class="as-side-brand">{ui.esc(res["name"].title())}</div>'
            f'<div class="as-muted">{res["confidence"] * 100:.1f}% confidence</div></div>'
            + ui.bars([(c.title(), p) for c, p in res["ranked"][:3]]),
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown('<p class="as-muted">No result yet.</p>', unsafe_allow_html=True)

    ui.sidebar_title("Model")
    metrics = ui.read_json(ui.ARTIFACTS_DIR / "crop_metrics.json", {}) or {}
    n_features = len(features) if hasattr(features, "__len__") else 7
    items = [("Algorithm", type(model).__name__, ""), ("Crops", len(le.classes_), ""), ("Inputs", n_features, "")]
    if "accuracy" in metrics:
        items.append(("Accuracy", f"{metrics['accuracy'] * 100:.1f}%", metrics.get("dataset", "")))
    st.sidebar.markdown(ui.kpis(items, compact=True), unsafe_allow_html=True)
    if "accuracy" not in metrics:
        st.sidebar.markdown(
            '<p class="as-small">To show accuracy here, add <code>artifacts/crop_metrics.json</code> with an '
            "<code>accuracy</code> value between 0 and 1.</p>",
            unsafe_allow_html=True,
        )


def render():
    ui.page_header(
        "Machine learning",
        "Crop recommendation",
        "Enter your soil and climate conditions. The model ranks the crops that fit them best and explains how "
        "your conditions compare with what it has seen.",
    )

    try:
        model, scaler, le, features = load_artifacts()
    except Exception as e:
        ui.sidebar_brand("Crop recommendation")
        st.error(
            "Could not load the crop model files. Make sure the `artifacts/` folder (scaler, label encoder, "
            f"features and model files) is next to `app.py`.\n\n`{e}`"
        )
        return

    with st.form("crop_form"):
        soil, climate = st.columns(2, gap="large")
        c1, c2 = st.columns(2)
        with soil:
            st.markdown("#### Soil")
            N = c1.number_input("Nitrogen (N)", 0, 200, 60, help="Nitrogen content in soil (mg/kg). Typical range 0 to 179.")
            P = c2.number_input("Phosphorus (P)", 0, 150, 50, help="Phosphorus content in soil (mg/kg). Typical range 5 to 145.")
            K = c1.number_input("Potassium (K)", 0, 210, 40, help="Potassium content in soil (mg/kg). Typical range 5 to 205.")
            ph = c2.number_input("Soil pH", 3.0, 14.0, 6.5, step=0.01, help="Typical range 3.5 to 9.9.")
        with climate:
            st.markdown("#### Climate")
            temperature = c1.number_input("Temperature (°C)", 0.0, 50.0, 25.0, step=0.1, help="Typical range 8 to 44 °C.")
            humidity = c2.number_input("Humidity (%)", 0.0, 100.0, 50.0, step=0.1, help="Typical range 0 to 100 %.")
            rainfall = c1.number_input("Rainfall (mm)", 20.0, 300.0, 100.0, step=0.5, help="Typical range 20 to 298 mm.")
        submitted = st.form_submit_button("Recommend crop", type="primary")

    if submitted:
        values = {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall}
        order = [N, P, K, temperature, humidity, ph, rainfall]
        st.session_state["crop_result"] = {"res": predict(model, scaler, le, order), "values": values}

    saved = st.session_state.get("crop_result")
    if saved:
        render_results(saved)
    else:
        render_welcome(le)
    render_sidebar(model, le, features, saved)
