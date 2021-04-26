
title = "My Awesome DataScientest project."
sidebar_name = "Introduction"


def tab():
    import streamlit as st

    st.title(title)

    st.markdown(
        """
        Here is a bootsrap template for your DataScientest project, built with [Streamlit](https://streamlit.io).

        You can browse streamlit documentation and demos to get some inspiration:
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into streamlit [documentation](https://docs.streamlit.io)
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset] (https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset]
          (https://github.com/streamlit/demo-uber-nyc-pickups)
        """
    )