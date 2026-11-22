"""Home page."""

import streamlit as st

import charts
import ui
from views import crop as crop_page
from views import disease as disease_page


def _crop_facts():
    try:
        _model, _scaler, le, features = crop_page.load_artifacts()
        return {"crops": sorted(str(c).title() for c in le.classes_),
                "inputs": len(features) if hasattr(features, "__len__") else 7}
    except Exception:
        return None


def _disease_facts():
    classes = ui.read_json(ui.DISEASE_DIR / "class_names.json")
    if not classes:
        return None
    crops = sorted({disease_page.split_label(c)[0] for c in classes})
    metrics = ui.read_json(ui.DISEASE_DIR / "metrics.json", {}) or {}
    return {"classes": classes, "crops": crops, "metrics": metrics}


def render():
    crops_info = _crop_facts()
    dis = _disease_facts()

    st.markdown(
        '<div class="as-eyebrow" style="margin-top: 1rem;">Agriculture Smart System</div>'
        '<h1 class="as-hero-title">From soil to leaf,<br>in one place.</h1>'
        '<p class="as-lede">Two machine learning tools for everyday farm decisions: choose the crop that suits '
        "your soil and climate, and check a leaf photo for disease.</p>"
        '<div class="as-rule"></div>',
        unsafe_allow_html=True,
    )

    # KPI row
    items = []
    if crops_info:
        items.append(("Crops recommended", len(crops_info["crops"]), "by the crop model"))
        items.append(("Soil and climate inputs", crops_info["inputs"], "N, P, K, pH, climate"))
    if dis:
        items.append(("Leaf conditions", len(dis["classes"]), f"across {len(dis['crops'])} crops"))
        if "test_accuracy" in dis["metrics"]:
            items.append(("Disease test accuracy", f"{dis['metrics']['test_accuracy'] * 100:.1f}%", "on held-out images"))
    if items:
        st.markdown(ui.kpis(items), unsafe_allow_html=True)

    # Tool cards
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="as-card-label">Tool 01 · Machine learning</div>'
                "<h3>Crop recommendation</h3>"
                '<p class="as-muted">Enter soil nutrients (nitrogen, phosphorus, potassium), pH, temperature, humidity '
                "and rainfall. You get a ranked list of the best suited crops, how your conditions compare with "
                "typical ranges, and growing notes.</p>",
                unsafe_allow_html=True,
            )
            if "crop" in ui.PAGES:
                st.page_link(ui.PAGES["crop"], label="Open crop recommendation", icon=":material/arrow_forward:")
    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="as-card-label">Tool 02 · Deep learning (CNN)</div>'
                "<h3>Leaf disease detection</h3>"
                '<p class="as-muted">Upload or photograph a single leaf. A lightweight image model names the crop and '
                "the condition, shows how sure it is, and links the result to a plain language guide with symptoms "
                "and management.</p>",
                unsafe_allow_html=True,
            )
            if "disease" in ui.PAGES:
                st.page_link(ui.PAGES["disease"], label="Open leaf disease detection", icon=":material/arrow_forward:")

    # How it works
    ui.section("How it works")
    a, b = st.columns(2, gap="large")
    with a:
        ui.card('<div class="as-card-label">Crop recommendation</div>' + ui.steps([
            ("Enter conditions.", "Soil and climate values for your farm."),
            ("Model scores every crop.", "A stacked ensemble of classifiers ranks the crops it knows."),
            ("Compare and decide.", "See the top options, a snapshot of your soil, and growing notes."),
        ]))
    with b:
        ui.card('<div class="as-card-label">Leaf disease detection</div>' + ui.steps([
            ("Add a leaf photo.", "Upload one or use the camera."),
            ("The CNN reads the leaf.", "The image is resized to 224 x 224 pixels and scored against every condition."),
            ("Check the result.", "Read the confidence, the alternatives and the condition guide."),
        ]))

    # Coverage and performance
    if dis:
        ui.section("What the disease model covers", "Crops and conditions it was trained to recognize.")
        c, d = st.columns([2, 3], gap="large")
        with c:
            groups = {}
            for raw in dis["classes"]:
                crop, cond, healthy = disease_page.split_label(raw)
                groups.setdefault(crop, []).append(cond)
            body = "".join(
                f'<p style="margin-bottom:0.4rem"><b>{ui.esc(crop)}</b></p>' + ui.plain_chips(sorted(conds))
                for crop, conds in sorted(groups.items())
            )
            ui.card(body)
        with d:
            reports = disease_page.load_reports()
            pc = reports.get("per_class")
            if pc is not None:
                with st.container(border=True):
                    st.markdown('<div class="as-card-label">Recall by condition on the test set</div>', unsafe_allow_html=True)
                    ui.show_fig(charts.recall_chart([disease_page.pretty(i) for i in pc.index], pc["recall"].values,
                                                    pc["recall_ci_low"].values, pc["recall_ci_high"].values))
            else:
                ui.card('<div class="as-card-label">Model performance</div><p class="as-muted">Add the report files to '
                        "<code>models/disease/reports/</code> to show a per condition chart here.</p>")

    if crops_info:
        ui.section("What the crop model recommends")
        ui.card(ui.plain_chips(crops_info["crops"]))

    # Notes
    ui.section("Good to know")
    n1, n2, n3 = st.columns(3, gap="large")
    notes = [
        ("For better photos", "Use one leaf, in good light, filling most of the frame. Plain backgrounds work best."),
        ("Confidence matters", "When the disease model is unsure it says so. Retake the photo instead of trusting a weak match."),
        ("A guide, not a diagnosis", "Confirm with a local agricultural extension officer before treating a crop."),
    ]
    for col, (title, text) in zip((n1, n2, n3), notes):
        with col:
            ui.card(f'<div class="as-card-label">{ui.esc(title)}</div><p>{ui.esc(text)}</p>')

    # Sidebar
    ui.sidebar_brand("Home")
    ui.sidebar_title("At a glance")
    side = []
    if crops_info:
        side.append(("Crops", len(crops_info["crops"]), ""))
    if dis:
        side.append(("Conditions", len(dis["classes"]), ""))
        if "test_accuracy" in dis["metrics"]:
            side.append(("Disease accuracy", f"{dis['metrics']['test_accuracy'] * 100:.1f}%", ""))
        if "test_macro_f1" in dis["metrics"]:
            side.append(("Macro F1", f"{dis['metrics']['test_macro_f1'] * 100:.1f}%", ""))
    if side:
        st.sidebar.markdown(ui.kpis(side, compact=True), unsafe_allow_html=True)
    ui.sidebar_title("Where to start")
    st.sidebar.markdown('<p class="as-muted">Choose a tool from the navigation bar. Each page shows the model '
                        "performance and your latest result in this sidebar.</p>", unsafe_allow_html=True)
