import streamlit as st
from streamlit_option_menu import option_menu
from models import  Home, ImageChat, Text2Image, PdfChat


# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "ImageChat", "PdfChat", "Text2Image"],  # required
                icons=["chat", "camera", "book","shuffle"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "ImageChat", "PdfChat", "Text2Image"],  # required
            icons=["chat", "camera", "book", "shuffle"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "ImageChat", "PdfChat", "Text2Image"],  # required
            icons=["chat", "camera", "book", "shuffle"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Home":
    Home.show_home()
if selected == "Text2Image":
    Text2Image.show_text2img()
if selected == "PdfChat":
    PdfChat.show_user_pdf()
if selected == "ImageChat":
    ImageChat.show_image()