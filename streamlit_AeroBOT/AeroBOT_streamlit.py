import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
import os
import gdown
import zipfile
import argparse
import pickle as pkl
import matplotlib.pyplot as plt

LOGGER = get_logger(__name__)

st.set_page_config(page_title="AeroBOT Demo",
                  page_icon="✈") #🛩
st.markdown('## ✈ Welcome to the AeroBOT streamlit demo!')
