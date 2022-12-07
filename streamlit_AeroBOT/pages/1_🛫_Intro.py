import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
import zipfile
import argparse
import pickle as pkl
import matplotlib.pyplot as plt

st.set_page_config(page_title="AeroBOT Demo",
                  page_icon="✈") #🛩

#   <!--- airplane image --->
#   <b title="Kiefer. from Frankfurt, Germany, CC BY-SA 2.0 &lt;https://creativecommons.org/licenses/by-sa/2.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Lufthansa_Airbus_A380_and_Boeing_747_(16431502906).jpg"><img width="500" alt="Lufthansa Airbus A380 and Boeing 747 (16431502906)" align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Lufthansa_Airbus_A380_and_Boeing_747_%2816431502906%29.jpg/512px-Lufthansa_Airbus_A380_and_Boeing_747_%2816431502906%29.jpg">
#     </b>
    
#   <sub><sub>
#     <a href="https://commons.wikimedia.org/wiki/File:Lufthansa_Airbus_A380_and_Boeing_747_(16431502906).jpg">Kiefer. from Frankfurt, Germany</a>, <a href="https://creativecommons.org/licenses/by-sa/2.0">CC BY-SA 2.0</a>, via Wikimedia Commons
#   </sub>


st.markdown(
  """
  # 🛫 Introduction
  This repository contains the code for our 6-month project **AeroBOT**, developed during our [Data Scientist training programe](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/) in 2022.


  ## Project overview

  **AeroBOT** is an automatic text classification project that tackles timely challenges in **Technical Language Processing (TLP)**, i.e. the domain-driven approach to using **Natural Language Processing (NLP)** in a **technical engineering context** with heavy presence of technical **jargon**. 
  <br/>The methodology developped in the project is transposable to industrial uses cases involving textual data in predictive maintenance, production, customer relationship, human resources, legal domain, to state a few.

  In **AeroBOT** we use approx. **100,000 labeled narratives** from **NASA**’s **Aviation Safety Reporting System (ASRS)** database, that describe **abnormal events** of the last 20 years in the **US airspace**.
  <br/>Our **objective** is to identify the most appropriate **target feature** in our dataset and **develop an algorithm** that correctly assigns labels to textual narratives. 

  We use a supervised approach for the **multiclass (x14), multiple-label** classification problem (more than 67% of the narratives have at least two different labels) with **imbalanced distribution** of labels (the most frequent label has ~30x higher occurrence compared to the least occuring one). 

  We compare the classification performance of **bag-of-word (BoW) models** (Decision Trees, Random Forest, Naive Bayes, SVM) combined with **preprocessing** of the data vs. **word embedding algorithms** vs. the **state-of-the-art transformer model [```BERT```](http://arxiv.org/abs/1810.04805)**, that we fine-tune, i.e. partially re-train on our data in a **Transfer Learning** context. 

  We compare the **1-vs-all** (14 models trained for 14 labels to be correctly assigned) vs. the **multilabel** approach (one model predicts all 14 labels for each narrative), the latter producing **versatile** models that are relatively **fast** to train (~1h for the retrained transformer model, on Google Colab with premium GPU).

  **Word embedding** models outperform BoW models and the retrained BERT-base model performs best, using raw data, with f1-scores ranging from **54% to 86%**, as measured on a final test set of ~10,000 entries, that was isolated in the beginning of the project. 

  **Partially retraining the BERT-base model on our data results in a performance increase of tens of percent, compared to the use of the ‘frozen’ BERT-base.**

  Our **threshold optimization algorithm** that boosts the f1-score of our transformer model by 1% to 5%, depending on the label and without necessitating any training. 

  Last but not least, we perform a **critical error analysis** by discussing the observed variations on the performance of our transformer model.

  *The program ```AeroBOT.py``` described below demonstrates the inference procedure of our BERT-based transformer model on the final test set of data.
  The rest of the content is found in the notebooks available on this repository.*
  """
)