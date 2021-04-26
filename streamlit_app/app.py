import inspect
import textwrap
from collections import OrderedDict

import streamlit as st
from streamlit.logger import get_logger
from tabs import intro, second_tab, third_tab


with open("style.css", "r") as f:
    style = f.read()

st.set_page_config(
    page_title="My Awesome App",
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png"
)

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

LOGGER = get_logger(__name__)

# Dictionary of
# demo_name -> (demo_function, demo_description)
DEMOS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (second_tab.sidebar_name, second_tab),
        (third_tab.sidebar_name, third_tab),
    ]
)


def run():
    st.sidebar.image("assets/logo-datascientest.png")
    demo_name = st.sidebar.radio("Menu", list(DEMOS.keys()), 0)
    demo = DEMOS[demo_name]

    demo.tab()


if __name__ == "__main__":
    run()
