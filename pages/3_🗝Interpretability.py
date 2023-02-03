import streamlit as st
import os
from pathlib import Path
import inspect
from streamlitpackages import get_img_with_href, get_image
from PIL import Image

# st.set_page_config(page_title="AeroBOT Demo",
#                   page_icon="✈")

# Configure sidebar
streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0])
with st.sidebar:
  st.header("Contact")
  logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/ioannis-stasinopoulos/', 20)
  st.write(f'''<a href="https://www.linkedin.com/in/ioannis-stasinopoulos/" style="text-decoration: none;color:black">Ioannis STASINOPOULOS</a> {logo_linkedin}''', unsafe_allow_html=True) 
  st.write(f'''<a href="https://www.linkedin.com/in/heleneassir/" style="text-decoration: none;color:black">Hélène ASSIR</a> {logo_linkedin}''', unsafe_allow_html=True) 

# 🌐 🎯 🧩 👓 🎓 🔎 🔦 🗝 🔓 🔭 ⁉ ❓ℹ ㊙ 🗣
st.markdown("""
          # 🗝 Interpretability
          ## bla bla
          """)
