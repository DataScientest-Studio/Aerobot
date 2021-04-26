title = "Third tab"
sidebar_name = "Third Tab"


def tab():
    import streamlit as st
    import pandas as pd
    import numpy as np

    st.title(title)

    st.markdown(
        """
        This is the third sample tab.
        """
    )

    st.write(pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD")))
