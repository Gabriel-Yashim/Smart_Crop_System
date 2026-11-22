"""Leaf disease detection page (CNN exported to ONNX, run with ONNX Runtime)."""

import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import charts
import knowledge as kb
import ui

HIGH_CONFIDENCE = 0.80   # at or above this, the prediction is shown as reliable
LOW_CONFIDENCE = 0.50    # below this, the app asks for a new photo instead of naming a disease


# ==== loading
@st.cache_resource(show_spinner=False)
def load_model():
    import onnxruntime as ort

    d = ui.DISEASE_DIR
    classes = json.loads((d / "class_names.json").read_text())
    cfg = json.loads((d / "preprocess.json").read_text())
    session = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])
    metrics = ui.read_json(d / "metrics.json", {}) or {}
    return {"session": session, "classes": classes, "cfg": cfg, "metrics": metrics}


def find_report(name):
    for folder in (ui.DISEASE_DIR / "reports", ui.DISEASE_DIR):
        path = folder / name
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_reports():
    out = {}
    p = find_report("test_per_class.csv")
    if p is not None:
        out["per_class"] = pd.read_csv(p, index_col=0)
    p = find_report("confidence_thresholds.csv")
    if p is not None:
        out["thresholds"] = pd.read_csv(p)
    p = find_report("top_confusions.csv")
    if p is not None:
        out["confusions"] = pd.read_csv(p)
    p = find_report("confusion_matrix.png")
    if p is not None:
        out["confusion_png"] = str(p)
    return out


# ==== helpers
def split_label(raw):
    """'Tomato__Tomato_Late_blight' -> ('Tomato', 'Late blight', False)."""
    crop, _, cond = raw.partition("__")
    if "_-_" in cond:
        cond = cond.split("_-_", 1)[1]
    low = cond.lower()
    healthy = "healthy" in low or low.startswith("fresh")
    if healthy:
        condition = "Healthy"
    else:
        for prefix in (crop + "_", crop.replace("_", " ") + " "):
            if cond.startswith(prefix):
                cond = cond[len(prefix):]
        condition = cond.replace("_", " ").strip()
        condition = condition[:1].upper() + condition[1:]
    return crop.replace("_", " "), condition, healthy


def pretty(raw):
    crop, cond, _ = split_label(raw)
    return f"{crop} · {cond}"


def preprocess(img, cfg):
    """Exactly the preprocessing used for validation and test images during training."""
    size = tuple(cfg.get("resize", [224, 224]))
    img = ImageOps.exif_transpose(img).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    arr = (arr - mean) / std
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


def analyze(bundle, img):
    """Returns every class ranked by probability: [(raw_class_name, probability), ...]."""
    cfg = bundle["cfg"]
    x = preprocess(img, cfg)
    logits = bundle["session"].run([cfg.get("output_name", "logits")], {cfg.get("input_name", "input"): x})[0][0]
    z = logits - logits.max()
    p = np.exp(z) / np.exp(z).sum()
    order = np.argsort(p)[::-1]
    return [(bundle["classes"][i], float(p[i])) for i in order]


def photo_checks(img):
    """Simple, rough checks that explain most bad results. They are a guide, not a rule."""
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    g = img.convert("L")
    target_w = min(256, w)   # never enlarge, which would look soft by itself
    g = g.resize((target_w, max(8, int(target_w * h / w))))
    a = np.asarray(g, dtype=np.float32)
    brightness = float(a.mean())
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    detail = float(lap.var())

    rows = [("Size", f"{w} x {h} px",
             ui.pill("Small", "warn") if min(w, h) < 224 else ui.pill("Good", "ok"))]
    if brightness < 60:
        rows.append(("Light", "Dark", ui.pill("Add light", "warn")))
    elif brightness > 205:
        rows.append(("Light", "Very bright", ui.pill("Avoid glare", "warn")))
    else:
        rows.append(("Light", "Balanced", ui.pill("Good", "ok")))
    if brightness >= 60:   # sharpness is not judged on dark photos, where low contrast looks like blur
        rows.append(("Detail", "Soft" if detail < 15 else "Sharp",
                     ui.pill("May be blurry", "warn") if detail < 15 else ui.pill("Good", "ok")))
    return rows


def level(conf):
    if conf >= HIGH_CONFIDENCE:
        return "High confidence", "ok"
    if conf >= LOW_CONFIDENCE:
        return "Moderate confidence", "warn"
    return "Low confidence", "bad"


# ==== result blocks
def render_result(ranked):
    raw, conf = ranked[0]
    crop, condition, healthy = split_label(raw)
    text, tone = level(conf)

    if conf < LOW_CONFIDENCE:
        ui.hero_card("Result", "Not sure about this one", ui.pill(text, tone), conf * 100)
        ui.card('<div class="as-card-label">What to try</div>'
                "<p>The model could not match this photo to a known condition reliably. Take a new photo of a single "
                "leaf in daylight, filling most of the frame, on a plain background. Make sure the crop is one the "
                "model knows (see the Supported crops tab).</p>")
    else:
        status = ui.pill("Healthy leaf", "ok") if healthy else ui.pill("Disease detected", "bad")
        ui.hero_card(crop, condition, status + ui.pill(text, tone), conf * 100)
        if conf < HIGH_CONFIDENCE:
            ui.card('<div class="as-card-label">Be careful</div><p>The model is only moderately sure. A clearer, '
                    "closer photo of the affected area will give a more reliable answer.</p>")

    with st.container(border=True):
        title = "Closest matches" if conf < LOW_CONFIDENCE else "Top 5 matches"
        st.markdown(f'<div class="as-card-label">{title}</div>', unsafe_allow_html=True)
        ui.show_fig(charts.hbar([(pretty(r), p) for r, p in ranked[:5]]))


def render_guide(bundle, ranked):
    classes = bundle["classes"]
    options = sorted(classes, key=lambda r: (split_label(r)[0], split_label(r)[1] != "Healthy", split_label(r)[1]))
    pred = ranked[0][0] if ranked and ranked[0][1] >= LOW_CONFIDENCE else None
    index = options.index(pred) if pred in options else 0
    choice = st.selectbox("Choose a crop and condition to read about", options, index=index,
                          format_func=pretty, key=f"guide_{pred}")
    crop, condition, healthy = split_label(choice)
    info = kb.condition_info(choice, healthy)
    if info is None:
        st.info("No guide is available for this condition yet.")
        return
    ui.card(
        f'<div class="as-card-label">{ui.esc(crop)}</div><h3>{ui.esc(condition)}</h3>'
        f'<p><b>What it is.</b> {ui.esc(info["about"])}</p>'
        f'<p><b>What to look for.</b> {ui.esc(info["symptoms"])}</p>'
        f'<p><b>What helps.</b> {ui.esc(info["management"])}</p>'
        '<p class="as-small">General information only. Confirm the diagnosis and the right treatment with a local '
        "agricultural extension officer.</p>"
    )


def render_breakdown(bundle, ranked):
    if not ranked:
        st.info("Add a photo to see how the model's confidence is spread across crops and conditions.")
        return
    by_crop = {}
    healthy_p = 0.0
    for raw, p in ranked:
        crop, _, healthy = split_label(raw)
        by_crop[crop] = by_crop.get(crop, 0.0) + p
        if healthy:
            healthy_p += p
    crops_sorted = sorted(by_crop.items(), key=lambda kv: -kv[1])

    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown('<div class="as-card-label">Which crop does it think this is</div>', unsafe_allow_html=True)
            ui.show_fig(charts.hbar(crops_sorted, xlabel="Share of confidence (%)"))
    with b:
        ui.card('<div class="as-card-label">Healthy or diseased</div>'
                + ui.bars([("Healthy leaf", healthy_p), ("Disease", 1 - healthy_p)] if healthy_p >= 0.5
                          else [("Disease", 1 - healthy_p), ("Healthy leaf", healthy_p)])
                + '<p class="as-small">Adds up the confidence of all healthy classes against all disease classes.</p>')
        top_crop, top_share = crops_sorted[0]
        if top_share < 0.8:
            ui.card('<div class="as-card-label">Mixed signal</div><p>The model splits its confidence across more than '
                    "one crop. That usually means the leaf is unusual, partly out of frame, or not one of the crops "
                    "it was trained on.</p>")
    with st.container(border=True):
        st.markdown('<div class="as-card-label">Every condition, ranked</div>', unsafe_allow_html=True)
        ui.show_fig(charts.hbar([(pretty(r), p) for r, p in ranked], highlight=1))


def render_performance(bundle):
    m = bundle["metrics"]
    reports = load_reports()
    if m:
        items = [("Test accuracy", f"{m['test_accuracy'] * 100:.1f}%", f"{m.get('n_test_images', 0):,} test images")]
        if "test_macro_f1" in m:
            ci = m.get("test_macro_f1_ci95")
            note = f"95% interval {ci[0] * 100:.1f} to {ci[1] * 100:.1f}" if ci else ""
            items.append(("Macro F1", f"{m['test_macro_f1'] * 100:.1f}%", note))
        if "test_weighted_f1" in m:
            items.append(("Weighted F1", f"{m['test_weighted_f1'] * 100:.1f}%", ""))
        if "onnx_cpu_latency_ms" in m:
            items.append(("Speed", f"{m['onnx_cpu_latency_ms']:.0f} ms", "per photo on a CPU"))
        if "onnx_size_mb" in m:
            items.append(("Model size", f"{m['onnx_size_mb']:.0f} MB", m.get("model", "")))
        st.markdown(ui.kpis(items), unsafe_allow_html=True)
    else:
        st.info("Add `metrics.json` to `models/disease/` to show test results here.")

    ui.card('<div class="as-card-label">Read this with care</div><p>These scores come from held-out images in the '
            "same dataset the model learned from. Photos taken in the field, with cluttered backgrounds or unusual "
            "lighting, can score lower. Macro F1 gives every condition equal weight, so it is the stricter number.</p>")

    pc = reports.get("per_class")
    if pc is not None:
        weakest = pc.sort_values("recall").head(2)
        sentences = [f"{pretty(i)} ({r.recall * 100:.0f}% recall on {int(r.support)} test images)" for i, r in weakest.iterrows()]
        a, b = st.columns([3, 2], gap="large")
        with a:
            with st.container(border=True):
                st.markdown('<div class="as-card-label">Recall by condition</div>', unsafe_allow_html=True)
                ui.show_fig(charts.recall_chart([pretty(i) for i in pc.index], pc["recall"].values,
                                                pc["recall_ci_low"].values, pc["recall_ci_high"].values))
        with b:
            ui.card('<div class="as-card-label">Weakest spots</div><p>The least reliable conditions are '
                    + " and ".join(ui.esc(s) for s in sentences)
                    + ". Small test sets make these figures less certain, which the wider bars show.</p>")
            conf = reports.get("confusions")
            if conf is not None and len(conf):
                top = conf.head(6)
                rows = "".join(
                    f"<tr><td>{ui.esc(pretty(r.true))}</td><td>{ui.esc(pretty(r.predicted))}</td><td>{int(r['count'])}</td></tr>"
                    for _, r in top.iterrows()
                )
                ui.card('<div class="as-card-label">Most common mix-ups</div>'
                        f'<table class="as-table"><tr><th>True</th><th>Predicted as</th><th>#</th></tr>{rows}</table>')

    th = reports.get("thresholds")
    if th is not None:
        c, d = st.columns([3, 2], gap="large")
        with c:
            with st.container(border=True):
                st.markdown('<div class="as-card-label">Confidence threshold trade-off</div>', unsafe_allow_html=True)
                ui.show_fig(charts.threshold_chart(th["threshold"].values, th["coverage"].values,
                                                   th["accuracy_on_accepted"].values))
        with d:
            ui.card('<div class="as-card-label">Why the app can say "not sure"</div><p>Asking for higher confidence '
                    "makes the answers it does give more accurate, but it answers fewer photos. This app asks for a "
                    "new photo below 50% confidence and warns below 80%.</p>")

    png = reports.get("confusion_png")
    if png:
        with st.expander("Confusion matrix on the test set"):
            st.image(png)
    if pc is None and th is None and not png:
        st.markdown('<p class="as-small">Copy the files from <code>artifacts/reports/&lt;run&gt;/</code> into '
                    "<code>models/disease/reports/</code> to show charts here.</p>", unsafe_allow_html=True)


def render_supported(bundle):
    groups = {}
    for raw in bundle["classes"]:
        crop, cond, healthy = split_label(raw)
        groups.setdefault(crop, []).append((cond, healthy))
    cols = st.columns(3, gap="large")
    for i, crop in enumerate(sorted(groups)):
        with cols[i % 3]:
            conds = sorted(groups[crop], key=lambda t: (not t[1], t[0]))
            chips = "".join(
                f'<span class="as-chip" style="{"background:#E3EAD3;color:#3F5A22;border-color:#C9D6AD" if h else ""}">{ui.esc(c)}</span>'
                for c, h in conds
            )
            ui.card(f'<div class="as-card-label">{ui.esc(crop)}</div>{chips}')
    st.markdown('<p class="as-small">Anything outside these crops will be forced into the closest known class, so '
                "check that the leaf belongs to one of them.</p>", unsafe_allow_html=True)


# ==== sidebar
def render_sidebar(bundle, ranked):
    ui.sidebar_brand("Leaf disease detection")
    m = bundle["metrics"]

    ui.sidebar_title("Model performance")
    if m:
        items = [("Accuracy", f"{m['test_accuracy'] * 100:.1f}%", ""), ("Macro F1", f"{m.get('test_macro_f1', 0) * 100:.1f}%", "")]
        items.append(("Test images", f"{m.get('n_test_images', 0):,}", ""))
        items.append(("Speed", f"{m.get('onnx_cpu_latency_ms', 0):.0f} ms", ""))
        st.sidebar.markdown(ui.kpis(items, compact=True), unsafe_allow_html=True)
        st.sidebar.markdown('<p class="as-small">Measured on held-out dataset images. Field photos can score lower.</p>',
                            unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="as-muted">No metrics file found.</p>', unsafe_allow_html=True)

    ui.sidebar_title("Latest result")
    if ranked:
        raw, conf = ranked[0]
        text, tone = level(conf)
        if conf < LOW_CONFIDENCE:
            head = '<div class="as-side-brand">Not sure</div>'
        else:
            crop, condition, _ = split_label(raw)
            head = (f'<div class="as-card-label">{ui.esc(crop)}</div>'
                    f'<div class="as-side-brand">{ui.esc(condition)}</div>')
        st.sidebar.markdown(
            f'<div class="as-card" style="margin-bottom:0.5rem">{head}{ui.pill(f"{text} ({conf * 100:.0f}%)", tone)}</div>'
            + ui.bars([(pretty(r), p) for r, p in ranked[:3]]),
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown('<p class="as-muted">No photo yet.</p>', unsafe_allow_html=True)

    ui.sidebar_title("How to read confidence")
    st.sidebar.markdown(
        f'<p class="as-muted">{ui.pill(f"{HIGH_CONFIDENCE * 100:.0f}% or more", "ok")} reliable<br>'
        f'{ui.pill(f"{LOW_CONFIDENCE * 100:.0f}% to {HIGH_CONFIDENCE * 100:.0f}%", "warn")} take care<br>'
        f'{ui.pill(f"Below {LOW_CONFIDENCE * 100:.0f}%", "bad")} retake the photo</p>',
        unsafe_allow_html=True,
    )


# ==== page
def render():
    ui.page_header(
        "Deep learning (CNN)",
        "Leaf disease detection",
        "Add a photo of a single leaf. The model names the crop and condition, shows how sure it is, and links the "
        "result to a plain language guide.",
    )

    try:
        bundle = load_model()
    except Exception as e:
        ui.sidebar_brand("Leaf disease detection")
        st.error(
            "Could not load the disease model. Put `model.onnx`, `class_names.json` and `preprocess.json` in "
            f"`models/disease/` next to `app.py`, and make sure `onnxruntime` is installed.\n\n`{e}`"
        )
        return

    left, right = st.columns([5, 6], gap="large")
    img, ranked = None, None

    with left:
        source = st.radio("Photo source", ["Upload a photo", "Use camera"], horizontal=True, label_visibility="collapsed")
        if source == "Upload a photo":
            file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        else:
            file = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")
        if file is not None:
            try:
                img = Image.open(file)
                img.load()
            except Exception:
                st.error("That file could not be read as an image. Please try another photo.")
                img = None
        if img is not None:
            st.image(ImageOps.exif_transpose(img), caption="Your photo")
            ui.card('<div class="as-card-label">Photo check</div>' + ui.kv_rows(photo_checks(img))
                    + '<p class="as-small">A rough guide to whether the photo is good enough.</p>')
        else:
            ui.card(
                '<div class="as-card-label">For a good result</div>'
                + ui.steps([
                    ("One leaf.", "Frame a single leaf, filling most of the picture."),
                    ("Good light.", "Use daylight and avoid harsh shadows or glare."),
                    ("Keep it sharp.", "Hold still and focus on the affected area."),
                    ("Plain background.", "A hand, paper or the ground behind the leaf works well."),
                ])
            )

    if img is not None:
        with st.spinner("Analyzing the leaf"):
            ranked = analyze(bundle, img)

    with right:
        if ranked:
            render_result(ranked)
        else:
            n_conditions = len(bundle["classes"])
            n_crops = len({split_label(c)[0] for c in bundle["classes"]})
            ui.card('<div class="as-card-label">Result</div><h3>Waiting for a photo</h3>'
                    '<p class="as-muted">Add a leaf photo on the left. The crop, the condition, the confidence and a '
                    "guide to the condition will appear here.</p>")
            st.markdown(ui.kpis([("Crops", n_crops, ""), ("Conditions", n_conditions, "")]), unsafe_allow_html=True)

    ui.section("Explore")
    t_guide, t_prob, t_perf, t_cov = st.tabs(["Condition guide", "Probability breakdown", "Model performance", "Supported crops"])
    with t_guide:
        render_guide(bundle, ranked)
    with t_prob:
        render_breakdown(bundle, ranked)
    with t_perf:
        render_performance(bundle)
    with t_cov:
        render_supported(bundle)

    render_sidebar(bundle, ranked)
