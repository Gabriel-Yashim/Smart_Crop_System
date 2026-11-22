"""Agriculture Smart System: Streamlit entry point.

Run locally:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Agriculture Smart System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

import ui  # noqa: E402
from views import crop, disease, home  # noqa: E402

ui.init_theme()

home_page = st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True)
crop_page = st.Page(crop.render, title="Crop recommendation", icon=":material/agriculture:", url_path="crop-recommendation")
disease_page = st.Page(disease.render, title="Leaf disease detection", icon=":material/eco:", url_path="disease-detection")

# Lets the home page link to the other two pages.
ui.PAGES["crop"] = crop_page
ui.PAGES["disease"] = disease_page

pages = [home_page, crop_page, disease_page]
try:
    nav = st.navigation(pages, position="top")   # needs a recent Streamlit
except TypeError:
    nav = st.navigation(pages)                   # older versions: sidebar navigation

nav.run()
ui.footer()
