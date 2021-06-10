from collections import OrderedDict

import streamlit as st
from streamlit.logger import get_logger

# TODO : you can add and rename tabs in the ./tabs folder,
# and import them here
from tabs import intro, second_tab, third_tab


with open("style.css", "r") as f:
    style = f.read()

st.set_page_config(
    page_title="My Awesome App",
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png"
)

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

LOGGER = get_logger(__name__)

# TODO: add new and/or renamed tab in this ordered dict
# passing the name in the sidebar as
DEMOS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (second_tab.sidebar_name, second_tab),
        (third_tab.sidebar_name, third_tab),
    ]
)


def run():
    st.sidebar.image(
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        width=200
    )
    demo_name = st.sidebar.radio("", list(DEMOS.keys()), 0)
    demo = DEMOS[demo_name]

    demo.tab()


if __name__ == "__main__":
    run()
