import streamlit as st
import os
from pathlib import Path
import inspect
from streamlitpackages import *

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
          ## Feature importance
          """)

# Plot the top 10 most important features for each Anomaly label
# Define choices for the user 
anomaly_tuple = (
    '01_Deviation / Discrepancy - Procedural',
    '02_Aircraft Equipment',
    '03_Conflict',
    '04_Inflight Event / Encounter',
    '05_ATC Issue',
    '06_Deviation - Altitude',
    '07_Deviation - Track / Heading',
    '08_Ground Event / Encounter',
    '09_Flight Deck / Cabin / Aircraft Event',
    '10_Ground Incursion',
    '11_Airspace Violation',
    '12_Deviation - Speed',
    '13_Ground Excursion',
    #'14_No Specific Anomaly Occurred'
    )

anomalies_to_plot = st.multiselect(
    'Choose the anomaly label(s) of interest:',
    anomaly_tuple,
    anomaly_tuple)

# LOAD token importance data
df_importances = load_df('df_importances_20230203.csv')

with st.spinner('Plotting...'):
  st.pyplot(plot_feature_importance(anomalies_to_plot, df_importances))