import streamlit as st
import os
from pathlib import Path
import inspect
import base64
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

st.set_page_config(page_title="AeroBOT Demo",
                  page_icon="✈") #🛩

st.markdown('# ✈ Welcome to the AeroBOT streamlit demo')

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

streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
photo = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/airplanes2.jpeg'), 'https://commons.wikimedia.org/wiki/File:Lufthansa_Airbus_A380_and_Boeing_747_(16431502906).jpg', 380)

st.markdown(
  """
  # 🛫 Introduction
  This  contains the code for our 6-month project **AeroBOT**, developed during our [Data Scientist training programe](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/) in 2022.

  ### *AeroBOT* project overview
  """)

st.markdown(photo, unsafe_allow_html=True) 
st.markdown('''<sub><sub>
            <a href="https://commons.wikimedia.org/wiki/File:Lufthansa_Airbus_A380_and_Boeing_747_(16431502906).jpg" style="text-decoration: none;color:gray">Kiefer. from Frankfurt, Germany</a>
            <a href="https://creativecommons.org/licenses/by-sa/2.0" style="text-decoration: none;color:gray">CC BY-SA 2.0, via Wikimedia Commons</a>
            ''', unsafe_allow_html=True)

st.markdown(
  """
  **AeroBOT** is an automatic text classification project that tackles timely challenges in **Technical Language Processing (TLP)**, i.e. the domain-driven approach to using **Natural Language Processing (NLP)** in a **technical engineering context** with heavy presence of technical **jargon**. 
  The methodology developped in the project is transposable to industrial uses cases involving textual data in predictive maintenance, production, customer relationship, human resources, legal domain, to state a few.

  In **AeroBOT** we use approx. **100,000 labeled narratives** from **NASA**’s **Aviation Safety Reporting System (ASRS)** database, that describe **abnormal events** of the last 20 years in the **US airspace**.
  Our **objective** is to identify the most appropriate **target feature** in our dataset and **develop an algorithm** that correctly assigns labels to textual narratives. 

  We use a supervised approach for the **multiclass (x14), multiple-label** classification problem (more than 67% of the narratives have at least two different labels) with **imbalanced distribution** of labels (the most frequent label has ~30x higher occurrence compared to the least occuring one). 

  We compare the classification performance of **bag-of-word (BoW) models** (Decision Trees, Random Forest, Naive Bayes, SVM) combined with **preprocessing** of the data vs. **word embedding algorithms** vs. the **state-of-the-art transformer model [```BERT```](http://arxiv.org/abs/1810.04805)**, that we fine-tune, i.e. partially re-train on our data in a **Transfer Learning** context. 

  We compare the **1-vs-all** (14 models trained for 14 labels to be correctly assigned) vs. the **multilabel** approach (one model predicts all 14 labels for each narrative), the latter producing **versatile** models that are relatively **fast** to train (~1h for the retrained transformer model, on Google Colab with premium GPU).

  **Word embedding** models outperform BoW models and the retrained BERT-base model performs best, using raw data, with f1-scores ranging from **54% to 86%**, as measured on a final test set of ~10,000 entries, that was isolated in the beginning of the project. 

  **Partially retraining the BERT-base model on our data results in a performance increase of tens of percent, compared to the use of the ‘frozen’ BERT-base.**

  Our **threshold optimization algorithm** that boosts the f1-score of our transformer model by 1% to 5%, depending on the label and without necessitating any training. 

  Last but not least, we perform a **critical error analysis** by discussing the observed variations on the performance of our transformer model.
  """
)

st.markdown(
  """
  ### Streamlit demo

  This demo addresses the following *fictitious* scenario: 

  Foreseeing budget cuts, NASA wishes to increase the efficiency of its activity related to its Aviation Safety Reporting System (ASRS). Their internal cost analysis has identified that the main operating costs are the expenditures to the analysts that NASA employs on the basis of temporary contracts for their competence in classifying the incoming reports of abnormal event events in the US airspace. Given their high professional qualification, these experts are costly and come with a very limited availability. Also, the job market is competitive and they tend to favor working for the private sector, where remuneration is higher compared to NASA’s offer.

  Our team at NASA’s internal Analytics Department, came up with a solution to this problem by minimizing the time these experts spend on the classification task. Reading the narratives is the most time-consuming part, hence we propose to develop a machine algorithm that captures the content of the texts and correctly classifies them according to the pre-defined ‘Anomaly’ labels in the ASRS database. This is a supervised classification problem, because we have access to thousands of previously labeled reports that we may use to train our algorithms. However, the labeling is not unique, since the same narrative can be tagged with two or more labels. 
  Our job is to find the most appropriate (probable) labels. 
  We are faced with an additional challenge: the data reveal an imbalanced distribution of labels: the most frequent label has ~30x higher occurrence compared to the least occuring one.

  Each classification expert who is employed, specializes in one label, or a family of labels, e.g. the expert with knowledge in Air Traffic Control (ATC) issues has limited knowledge in aircraft equipment matters.
  The expert enters the ASRS database and filters the results of our algorithm: they choose to visualize all the narratives that have been tagged according to their expertise. For example, if you are the expert on ATC, you select all narratives that have been labeled with the ‘ATC Issue’ anomaly. This means that they either have a ‘1’ in that column or that the probability for the ‘ATC Issue’ label, that has been estimated by the model, exceeds a certain threshold defined by NASA, e.g. 0.5.

  This demo shows how we evaluate the performance of our models.

  """)
