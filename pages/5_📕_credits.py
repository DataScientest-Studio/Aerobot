import streamlit as st 
import os
import base64
import streamlit.components.v1 as components
import inspect
from pathlib import Path   
import sys
from PIL import Image
streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))#.parents[0]
)
sys.path.append(streamlit_home_dir)
from streamlitpackages import get_img_with_href              

img_planes_path = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0].joinpath('ressources/airplanes2.jpeg')
image = Image.open(img_planes_path)

# Configure sidebar
streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0])
with st.sidebar:
  st.image(image)



st.markdown("# 📕 Credits")

def credits():
    streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0])

    logo_dataScientest = get_img_with_href(os.path.join(str(Path(streamlit_home_dir)), 'ressources/datascientestLogo_round.png'), 'https://datascientest.com/')

    # c1, c2 = st.columns([0.5, 2])
    # c1.
    st.markdown(logo_dataScientest, unsafe_allow_html=True) 
    # c2.
    st.markdown(f'''The project AeroBOT is a capstone project of a Data Scientist bootcamp at <a href="https://datascientest.com">DataScientest.com</a>''', unsafe_allow_html=True)
    st.write("")
    c1, c2, c3, c4  = st.columns([0.78, 1.2, 0.8, 1])
    with c1:
        st.markdown(f'''<b>Project members</b> :''', unsafe_allow_html=True)  
    with c2:
        logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/ioannis-stasinopoulos/', 20)
        st.markdown(f'''<a href="https://www.linkedin.com/in/ioannis-stasinopoulos/" style="text-decoration: none;color:black">Ioannis STASINOPOULOS</a> {logo_linkedin}''', unsafe_allow_html=True) 
    with c3:
        logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/heleneassir/', 20)
        st.markdown(f'''<a href="https://www.linkedin.com/in/heleneassir/" style="text-decoration: none;color:black">Hélène ASSIR</a> {logo_linkedin}''', unsafe_allow_html=True) 
    with c4:
        logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/hicheme-hadji-97294368/', 20)
        st.markdown(f'''<a href="https://www.linkedin.com/in/hicheme-hadji-97294368/" style="text-decoration: none;color:black">Hicheme HADJI</a> {logo_linkedin}''', unsafe_allow_html=True) 
    

    c1, c2 = st.columns([0.5, 2])
    c1.markdown(f'''<b>Project mentor</b> :''', unsafe_allow_html=True)  
    logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/alban-thuet-683365173/', 20)
    c2.markdown(f'''<a href="https://www.linkedin.com/in/alban-thuet-683365173/" style="text-decoration: none;color:black">Alban THUET (DataScientest)</a> {logo_linkedin}''', unsafe_allow_html=True)    

    c1, c2 = st.columns([0.5, 2])
    c1.markdown(f'''<b>Github</b> :''', unsafe_allow_html=True)  
    c2.markdown(f'''<a href="https://github.com/DataScientest-Studio/Aerobot">AeroBOT project</a>''', unsafe_allow_html=True) 

    c1, c2 = st.columns([0.5, 2])
    c1.markdown(f'''<b>Data source</b> :''', unsafe_allow_html=True)  
    c2.markdown(f'''<a href="https://asrs.arc.nasa.gov/">NASA - Aviation Safety Reporting System (ASRS) database</a>''', unsafe_allow_html=True) 

credits()