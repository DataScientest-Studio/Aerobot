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

st.markdown("""
          # 💡 Key learnings
          """)

st.markdown("""
We summarize key learnings from the Model Evaluation:

- First, we see that the **unsupervised BoW approach often underperforms** compared to the baseline model, especially Random Forests because of their feature sampling. Overall **TF-IDF remains the best option** of this approach subject to limiting the reduction of vocabulary
- Conversely, Random Forests will really benefit from the supervised feature selection of approach 3, with the intuition that a rare and specific vocabulary is essential for anomaly prediction.
- **Undersampling** provides f1-scores comparable to standard sampling, but with much higher recall at the expense of precision.
- **Optimizing hyper-parameters** improves performance only slightly but is essential for Gradient Boosting, which is very sensitive to incorrect initialization of its parameters. SVM are very greedy but have an equivalent performance in our use case.
- Finally, the **1vsAll mode** (1 model trained for each anomaly) allows us to do “cherry-picking” by keeping the best model for each anomaly that we summarized under “Best model” in its category. 
- Regarding **WordEmbedding**, the 1vsAll modelling shows the best performance, the other multilabel attempts being generally unsuccessful from the point of view of versatility.
- Finally for the **BERT approach**, a real split between the completely unsuccessful “Frozen” models and the “Unfrozen” models which will show the best performance of all approaches.
""")
