import streamlit as st 
import os
import base64
import streamlit.components.v1 as components
import inspect
from pathlib import Path                 

st.set_page_config(page_title="AeroBOT Demo",
                  page_icon="✈") #🛩
st.markdown("# 📕 Credits")

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url, size=50):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" height={size}px/>
        </a>'''
    return html_code


def credits():

    streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0])

    logo_dataScientest = get_img_with_href(os.path.join(str(Path(streamlit_home_dir)), 'ressources/datascientestLogo.png'), 'https://datascientest.com/')

    c1, c2 = st.columns([0.5, 2])
    c1.markdown(logo_dataScientest, unsafe_allow_html=True) 
    c2.markdown(f'''The project AeroBOT is a capstone project of a Data Scientist bootcamp at <a href="https://datascientest.com">DataScientest.com</a>''', unsafe_allow_html=True)
    st.write("")
    c1, c2, c3, c4  = st.columns([0.5, 1.2, 1, 1])
    with c1:
        st.markdown(f'''<b>Project members</b> :''', unsafe_allow_html=True)  
    with c2:
        logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/ioannis-stasinopoulos/', 20)
        st.markdown(f'''<a href="https://www.linkedin.com/in/ioannis-stasinopoulos/" style="text-decoration: none;color:white">Ioannis STASINOPOULOS</a> {logo_linkedin}''', unsafe_allow_html=True) 
    with c3:
        logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/heleneassir/', 20)
        st.markdown(f'''<a href="https://www.linkedin.com/in/heleneassir/" style="text-decoration: none;color:white">Hélène ASSIR</a> {logo_linkedin}''', unsafe_allow_html=True) 
    with c4:
        logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/hicheme-hadji-97294368/', 20)
        st.markdown(f'''<a href="https://www.linkedin.com/in/hicheme-hadji-97294368/" style="text-decoration: none;color:white">Hicheme HADJI</a> {logo_linkedin}''', unsafe_allow_html=True) 
    

    c1, c2 = st.columns([0.5, 2])
    c1.markdown(f'''<b>Project mentor</b> :''', unsafe_allow_html=True)  
    logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/alban-thuet-683365173/', 20)
    c2.markdown(f'''<a href="https://www.linkedin.com/in/alban-thuet-683365173/" style="text-decoration: none;color:white">Alban THUET (DataScientest)</a> {logo_linkedin}''', unsafe_allow_html=True)    

    c1, c2 = st.columns([0.5, 2])
    c1.markdown(f'''<b>Github</b> :''', unsafe_allow_html=True)  
    c2.markdown(f'''<a href="https://github.com/DataScientest-Studio/Aerobot">AeroBOT project</a>''', unsafe_allow_html=True) 

    c1, c2 = st.columns([0.5, 2])
    c1.markdown(f'''<b>Data source</b> :''', unsafe_allow_html=True)  
    c2.markdown(f'''<a href="https://asrs.arc.nasa.gov/">NASA - Aviation Safety Reporting System (ASRS) database</a>''', unsafe_allow_html=True) 

credits()