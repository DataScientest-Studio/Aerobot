import streamlit as st
import os
from pathlib import Path
import inspect
from streamlitpackages import get_img_with_href, get_image
from PIL import Image

# Configure sidebar
streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0])
with st.sidebar:
  st.header("Contact")
  logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/ioannis-stasinopoulos/', 20)
  st.write(f'''<a href="https://www.linkedin.com/in/ioannis-stasinopoulos/" style="text-decoration: none;color:black">Ioannis STASINOPOULOS</a> {logo_linkedin}''', unsafe_allow_html=True) 
  st.write(f'''<a href="https://www.linkedin.com/in/heleneassir/" style="text-decoration: none;color:black">Hélène ASSIR</a> {logo_linkedin}''', unsafe_allow_html=True) 

st.markdown("""
          # 🧪 Experiment protocol""")


st.image(get_image(img_name = 'Model_table.png'), 
        caption='Table summarizing our experimental protocol for each of the 5 main modelling approaches. \
        Arrows in the vocabulary column denote that the vocabulary size was reduced down to the respective minimum value shown.')

st.markdown("""
          We use a decision tree as the **baseline model**.
          We start with NLP preprocessed narratives and use a Countvectorizer without feature selection, nor undersampling. This yields a vocabulary of more than 50,000 tokens, which strongly limits our choice of classifiers. 
          We chose the Decision Tree because it’s a lightweight model with an embedded feature selection.
          The target variable is coded in the 1vs All mode, so 14 variables.
          
In the **second approach**, still in the Bag of words framework, we seek to reduce the size of the vocabulary, in an unsupervised way.Here, we experiment 2 ways to vectorize the TLP-preprocessed narratives:- Countvectorizer, which simply counts the word frequency in the narratives- and TF-IDF vectorizer, which also takes into account the specificity of the vocabulary in the corpus.Different vocabulary sizes down to 500 tokens are tested. The vocabulary size reduction allows us to train more consuming better-performing classifiers such as Random Forests or naive Bayesian classifiers. 

In the **third approach**, still in Bag of words, we reduce the size of the vocabulary even more , down to 20 tokens, by selecting the most important tokens as presented in the previous slide. This is why we call this “supervised feature selection”.Here we can test more demanding classifiers: Random Forest, Gradient Boosting and SVM, and also optimize their hyperparameters!The vocabulary is supervised for each anomaly, therefore necessarily in 1 vs. All mode, which allows us to test undersampling.

In the **fourth approach**, we leave the Bag of Words to represent the words by Word Embedding.With the dimension of the vectors representing each word, we switch to dense neural networks and the target feature is coded as one multilabel variable. 

In the **fifth approach**, we experiment with a Transformer model consisting of the 12 embedding layers of a BERT instance and “classification” layers in a dense neural network.We confronted two Transfer Learning approaches: ‘frozen’ BERT, keeping all the weights of the pre-trained model, vs ‘unfrozen’ where we re-train only the last 4 layers of BERT on our data.In this approach, the target feature is coded as one multilabel variable.

Considering all the modelling options tested, we end up with a total of a **hundred models**.
          """)
