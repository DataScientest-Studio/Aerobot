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

st.markdown("""
We use a decision tree as the baseline model, which has the advantage of interpretability. For each token in the vocabulary, it returns its importance for the prediction task.

Beyond interpretability, we also use the token importance to perform supervised feature selection for BoW models.
""")

# Plot the top 10 most important features for each Anomaly label
# Define choices for the user 
anomaly_tuple = st.session_state['anomaly_tuple'] # see main page

anomalies_to_plot = st.multiselect(
    'Choose the anomaly label(s) of interest:',
    anomaly_tuple,
    anomaly_tuple)

# LOAD token importance data
df_importances = load_df('df_importances_20230203.csv')

with st.spinner('Plotting...'):
  st.pyplot(plot_feature_importance(anomalies_to_plot, df_importances))

st.markdown(""" ### Our Analysis
As the baseline model is applied on NLP-preprocessed narratives, we can find e.g. in the ‘Anomaly_Aircraft Equipement’, both the full-text and the abbreviated forms of the same words such as:

- “maintenance”  with “mainten” as the full-text stemmed token and “maint” as the abbreviation (not stemmed, as it is not a common word)

- “emergency”  with “emerg” as the full-text stemmed token and “emer” as the abbreviation (not stemmed)

This gives us insight on the connection between our target features and the narratives, here represented by their stemmed tokens:

- **Deviation / Discrepancy**: discussion (declar, realiz, told, ask),  Traffic Collision Avoidance System (tcas), error
- **Aircraft Equipment**: maintenance / inspection,  Quick Reference Handbook (qrhf), aircraft parts (electr, engine)
- **Conflict**: Resolution advisory (ra=indication given to the flight crew recommending a maneuver), collision
- **Inflight Event / Encounter**: weather elements (turbulence, cloud, wind, mountain wave activity (mwa) or message waring area), terrain (probably proximity), approach, fuel
- **ATC Issue**: traffic-related keywords (traffic, sector, clearance, control, separation), ra (possible meanings)
- **Deviation - Altitude**: altitude, ft (feet), descend, 
- **Deviation - Track/Heading**: course, heading, turn, flight management system (fms)
- **Ground Event / Encounter**: damage, brake, structure, grass, propeller, ramp, gear, tug, rudder
- **Flight Deck / Cabin / Aircraft event**: smell, odor, smoke, fume, attend, medical, paramedical, passenger, ATC, pax (number of persons on board)
- **Ground Incursion**: taxiway (txwi), hold & short (hold short)
- **Airspace Violation**: airspace, violation, sector, traffic, enter, feet (ft), temporary flight restrictions (tfr), air defense identification zone (adiz), squawk
- **Deviation - Speed**: speed (spd), airspeed, overspeed, knots (kts), exceed, shaker, 250 (refers to the speed of 250 kts, the universal limit below a 10,000ft altitude)
- **Ground Excursion**: grass, field, mud, ditch, veer-off, damage, brake, runway (rwi)
""")