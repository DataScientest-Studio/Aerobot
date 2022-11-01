
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
<!---
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
<![GitHub Actions](https://img.shields.io/badge/githubactions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)>
-->


# Aerobot
<!--- airplane image --->

<a href="url"><img src="https://user-images.githubusercontent.com/97918270/167825700-7ed773a3-8088-4adb-9c81-3bed1f3a10a4.png" align="left" height="248" ></a>


This repository contains the code for our 6-month project **AeroBOT**, developed during our [Data Scientist training programe](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/) in 2022.

## Project overview

**AeroBOT** is an automatic text classification project that tackles timely challenges in **Technical Language Processing (TLP)**, i.e. the domain-driven approach to using **Natural Language Processing (NLP)** in a **technical engineering context** with heavy presence of technical **jargon**. 

We use approx. **100,000 labeled narratives** from **NASA**’s **Aviation Safety Reporting System (ASRS)** database, that describe **abnormal events** in the last 20 years in the **US airspace**. **Our goal** is to identify the most appropriate **target feature** in our dataset and **develop an algorithm** that correctly assigns labels to textual narratives. 

We use a supervised approach for the **multiclass (x14), multiple-label** classification problem (more than 67% of the narratives have at least two different labels) with **imbalanced distribution** of labels (the most frequent label has ~30x higher occurrence compared to the least occuring one). 

We compare the classification performance of **bag-of-word (BoW) models** (Decision Trees, Random Forest, Naive Bayes, SVM) combined with **preprocessing** of the data vs. **word embedding algorithms** vs. the **state-of-the-art transformer model ```BERT```**, that we fine-tune, i.e. partially re-train on our data in a **Transfer Learning** context. 

We compare the **1-vs-all** (14 models trained for 14 labels to be correctly assigned) vs. the **multilabel** approach (one model predicts all 14 labels for each narrative), the latter producing **versatile** models that are relatively **fast** to train (~1h for the retrained transformer model, on Google Colab with premium GPU).

**Word embedding** models outperform BoW models and the retrained BERT-base model performs best, using raw data, with f1-scores ranging from **54% to 86%**, as measured on a final test set of ~10,000 entries, that was isolated in the beginning of the project. 

**We observe that partially retraining the BERT-base model on our data results in a performance increase of tens of percent, compared to the use of the ‘frozen’ BERT-base.**

We present a **threshold optimization algorithm** that boosts the f1-score of our transformer model by 1% to 5%, depending on the label and without necessitating any training. 

Last but not least, we perform a **critical error analysis** by discussing the observed variations on the performance of our transformer model.

*The program ```AeroBOT.py``` described below demonstrates the inference procedure of our BERT-based transformer model on the final test set of data.
The rest of the content is found in the notebooks available on this repository.*

## Team
This project was developed by the following team :

- Ioannis STASINOPOULOS ([GitHub](https://github.com/Cochonaki)) / [LinkedIn](https://www.linkedin.com/in/ioannis-stasinopoulos/))

## Notebooks
You can browse though the [notebooks](./notebooks). 

## Data
The [data](./data)

## AeroBOT.py Program

### Installation

#### How to clone this repository
Unless you want to stay up-to-date with any changes in the repo by using git, you can also simply *copy* the repo contents to a local folder (see 'Code' button > Download zip) and ignore the following steps.
1. In your terminal, use ```cd``` to move to the directory, in which you wish to clone the repo.
2. [Generate a personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) from from your GitHUB account (google is your friend).
3. Execute the following lines in your terminal to clone the repo locally.
```
username='DataScientest-Studio'
repository='Aerobot'
git_token='YOUR TOKEN'
git clone https://"$git_token"@github.com/"$username"/"$repository"
```
#### How to execute the AeroBOT.py program in a (mini)conda environment
Pre-requisites: 
- [Install (mini)conda](https://docs.conda.io/en/latest/miniconda.html)
- The present Aerbot repository is cloned or copied locally (see instructions above)

1. Create a conda environment named e.g. 'aerobot_venv'
```
conda create --name aerobot_venv python=3.7.5
```
This installs python v3.7.5 and basic packages, e.g. pip, necessary to install the project packages later on.

2. Activate the environment
```
conda activate aerobot_venv
```

3. Navigate to the local mirror of the Aerobot repo using ```cd``` 
and install the necessary dependencies packages into your conda environment:
```
pip install -r requirements.txt
```
This takes several minutes.

Check the installation by listing the installed packages in 'aerobot_venv'.
```
conda list 
```
4. Execute the ```AeroBOT.py``` program
(make sure you have ```cd``` to the location of ```AeroBOT.py```)
```
python AeroBOT.py 100
```
This will infer the transformer model on the first 100 entries of the final test set.
If necessary, run
```
python AeroBOT.py --help
```
for help.
After successfull execution, you will see the classification report printed in the terminal and ```.pkl``` files created in ```./data/models/```
If you wish, you may now delete the repository from your computer, and deactivate and delete the conda environment:
```
conda deactivate
conda env remove --name aerobot_venv
```
Verify the operation by listing your conda environments
```
conda env list
```
___Note:___ The ```requirements_for_Google_Colab.txt``` is for installing dependencies on Google Colab. The file was generated with 
```
! pip freeze > requirements.txt
```
on Google Colab and contains an exhaustive list of the necessary packages.
Note that a "!" has to preceed all bash commands in Google Colab.

<!---
template d'application [Streamlit](https://streamlit.io/)
## Streamlit App - WORK in progress

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

**Docker**

You can also run the Streamlit app in a [Docker](https://www.docker.com/) container. To do so, you will first need to build the Docker image :

```shell
cd streamlit_app
docker build -t streamlit-app .
```

You can then run the container using :

```shell
docker run --name streamlit-app -p 8501:8501 streamlit-app
```

And again, the app should then be available at [localhost:8501](http://localhost:8501).
--->
