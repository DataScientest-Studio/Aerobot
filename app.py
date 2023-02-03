import streamlit as st
st.set_page_config( # `set_page_config()` can only be called once per app, and must be called as the first Streamlit command in your script.
    page_title="AeroBOT demo",
    page_icon="✈",  #🛩
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "https://github.com/DataScientest-Studio/Aerobot/issues",
        'About': "AeroBOT is an automatic text classification project. It is also the 6-month capstone project of our Data Scientist bootcamp at DataScientest in 2022."
    }
)
import os
from pathlib import Path
import inspect
from streamlitpackages import get_img_with_href

# Configure sidebar
streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))#.parents[0]
)
with st.sidebar:
  st.header("Contact")
  logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/ioannis-stasinopoulos/', 20)
  st.write(f'''<a href="https://www.linkedin.com/in/ioannis-stasinopoulos/" style="text-decoration: none;color:black">Ioannis STASINOPOULOS</a> {logo_linkedin}''', unsafe_allow_html=True) 
  st.write(f'''<a href="https://www.linkedin.com/in/heleneassir/" style="text-decoration: none;color:black">Hélène ASSIR</a> {logo_linkedin}''', unsafe_allow_html=True) 


st.markdown('# Welcome to the AeroBOT demo ✈')
photo = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/airplanes2.jpeg'), 'https://commons.wikimedia.org/wiki/File:Lufthansa_Airbus_A380_and_Boeing_747_(16431502906).jpg', 380)
st.markdown(
  """
  ## 🛫 Introduction
  **AeroBOT** is an automatic text classification project. 
  It is also the 6-month capstone project of our [Data Scientist bootcamp](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/) in 2022.
  """)

st.markdown(photo, unsafe_allow_html=True) 
st.markdown('''<sub><sub>
            <a href="https://commons.wikimedia.org/wiki/File:Lufthansa_Airbus_A380_and_Boeing_747_(16431502906).jpg" style="text-decoration: none;color:gray">Kiefer. from Frankfurt, Germany</a>
            <a href="https://creativecommons.org/licenses/by-sa/2.0" style="text-decoration: none;color:gray">CC BY-SA 2.0, via Wikimedia Commons</a>
            ''', unsafe_allow_html=True)

st.markdown(
  """
  **AeroBOT** tackles timely challenges in **Technical Language Processing (TLP)**, i.e. the domain-driven approach to using **Natural Language Processing (NLP)** in a **technical engineering context** with heavy presence of technical **jargon**. 
  The methodology developped in the project is transposable to industrial uses cases involving textual data in predictive maintenance, production, customer relationship, human resources, legal domain, to state a few.

  In **AeroBOT** we use approx. **100,000 labeled narratives** from **NASA**’s
  [**Aviation Safety Reporting System (ASRS)**](https://asrs.arc.nasa.gov/) database. 
  The narratives describe **abnormal events** of the last 20 years in the **US airspace**.
  
  Our **objective** is to **develop an algorithm** that correctly assigns 'anomaly' labels to these textual narratives. 

  This represents a **supervised, multiclass (x14), multiple-label** classification problem, because more than 67% of the narratives have at least two different labels.
  The labels also show an **imbalanced distribution**: the most frequent label has ~30x higher occurrence compared to the least occuring one. 

  We compare the classification performance of **bag-of-words (BoW) models** (Decision Trees, Random Forest, Naive Bayes, SVM) combined with **preprocessing** of the data vs. **word embedding algorithms** vs. the **state-of-the-art transformer model [BERT](http://arxiv.org/abs/1810.04805)**.
  
  We fine-tune, that is, we partially re-train the transformer model on our data in a **Transfer Learning** context. 

  #### Main results:
  - **Word embedding** models outperform BoW models and the retrained **BERT-base** model performs best, using totally unprocessed textual data, with f1-scores ranging from **54% to 86%**, as measured on a final test set of ~10,000 entries that was isolated in the beginning of the project. 

  - **Partially retraining the BERT-base model on our data results in a performance increase of tens of percent, compared to the use of the ‘frozen’ BERT-base.**
  """
)

st.markdown(
  """
  ## This demo

  The present demo addresses the following *fictitious* scenario: 

  Foreseeing budget cuts, NASA wishes to increase the efficiency of its activity related to its Aviation Safety Reporting System (ASRS). Their internal cost analysis has identified that the main operating costs are the expenditures to the analysts that NASA employs on the basis of temporary contracts for their competence in classifying the incoming reports of abnormal event events in the US airspace. Given their high professional qualification, these experts are costly and come with a very limited availability. Also, the job market is competitive and they tend to favor working for the private sector, where remuneration is higher compared to NASA’s offer.

  Our team at NASA’s internal Analytics Department came up with a solution to this problem by minimizing the time these experts spend on the classification task. Reading the narratives is the most time-consuming part, hence we propose to develop a machine algorithm that captures the content of the texts and correctly classifies them according to the pre-defined ‘Anomaly’ labels in the ASRS database.  
  
  The job of this algorithm is then to find the most appropriate (probable) labels for any new unknown narrative it is presented with.
  The chosen labels will determine which experts should review the narrative.
  
  This demo 
  - presents key aspects of our data set (page 'Data')
  - shows how we preprocess the narratives before feeding them to the Bag-of-words models (page 'Preprocessing')
  - details how we evaluate the performance of our models (page 'Model evaluation')
    - for a given label
    - across all labels
  - provides keys for interpreting the outputs of our baseline model (a Decision Tree)
  """)
