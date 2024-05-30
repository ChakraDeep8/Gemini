import os

import streamlit as st
from streamlit_navigation_bar import st_navbar

from pages import Home, ImageChat, Text2Image, PdfChat

st.set_page_config(initial_sidebar_state="collapsed")

pages = ["Home", "ImageChat", "PdfChat", "Text2Image"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
# logo_path = os.path.join(parent_dir, "cubes.svg")
urls = {"GitHub": st.secrets["github"]}
styles = {
    "nav": {
        "background-color": "royalblue",
        "justify-content": "left",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "white",
        "padding": "14px",
    },
    "active": {
        "background-color": "white",
        "color": "var(--text-color)",
        "font-weight": "normal",
        "padding": "14px",
    }
}
options = {
    "show_menu": False,
    "show_sidebar": False,
}

page = st_navbar(
    pages,
    # logo_path=logo_path,
    urls=urls,
    styles=styles,
    options=options,
)

functions = {
    "Home": Home.show_home,
    "ImageChat": ImageChat.show_image,
    "PdfChat": PdfChat.show_user_pdf,
    "Text2Image": Text2Image.show_text2img,
}
go_to = functions.get(page)
if go_to:
    go_to()
