![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/DataScientest-Studio/Aerobot/yourApp/)

<!--- ![HF](https://user-images.githubusercontent.com/97918270/199447402-9d02f298-c4a9-4e98-bf7f-5155658adac8.png) -->
<!---
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
<![GitHub Actions](https://img.shields.io/badge/githubactions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)>
-->


# Aerobot
This repository contains the code for our 6-month project **AeroBOT**, developed during our [Data Scientist training programe](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/) in 2022.

<!--- airplane image --->
<b title="Kiefer. from Frankfurt, Germany, CC BY-SA 2.0 &lt;https://creativecommons.org/licenses/by-sa/2.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Lufthansa_Airbus_A380_and_Boeing_747_(16431502906).jpg"><img width="500" alt="Lufthansa Airbus A380 and Boeing 747 (16431502906)" align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Lufthansa_Airbus_A380_and_Boeing_747_%2816431502906%29.jpg/512px-Lufthansa_Airbus_A380_and_Boeing_747_%2816431502906%29.jpg">
  </b>
  
<sub><sub>
  <a href="https://commons.wikimedia.org/wiki/File:Lufthansa_Airbus_A380_and_Boeing_747_(16431502906).jpg">Kiefer. from Frankfurt, Germany</a>, <a href="https://creativecommons.org/licenses/by-sa/2.0">CC BY-SA 2.0</a>, via Wikimedia Commons
</sub>


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

## Notebooks
The table below summarizes the ```.ipynb``` notebooks in AeroBOT (liks point to the files in the [notebooks folder](./notebooks)) and the diagram illustrates both their mutual dependencies and the flow of data. 
  
| Notebook Name                                                                                                                                                                                                                                                                                 | Description                                                                                                                      | Authors    |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| [01\_1\_Data\_Exploration\_and\_Dataviz\_part1](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/01_1_Data_Exploration_and_Dataviz_part1.ipynb)                                                                                                                       | Preliminary data exploration                                                                                                     | IS, HA     |
| [01\_2 Data Exploration and Dataviz part2](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/01_2%20Data%20Exploration%20and%20Dataviz%20part2.ipynb)                                                                                                                  | Preliminary data exploration                                                                                                     | HH         |
| [02 Test set creator](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/02%20Test%20set%20creator.ipynb)                                                                                                                                                               | Split original data into train / final test data sets                                                                            | IS, HA     |
| [03\_1 Narratives RegEx preprocessing](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/03_1%20Narratives%20RegEx%20preprocessing.ipynb)                                                                                                                              | TLP preprocessing of narratives using RegEx                                                                                      | IS, HA     |
| [03\_2 Narratives Standard text processing and Abbreviations](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/03_2%20Narratives%20Standard%20text%20processing%20and%20Abbreviations.ipynb)                                                                          | NLP + TLP preprocessing of narratives                                                                                            | IS, HA     |
| [03\_3 A trip from the Bahamas Example Narrative preprocessing](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/03_3%20A%20trip%20from%20the%20Bahamas%20Example%20Narrative%20preprocessing.ipynb)                                                                  | Narrative preprocessing example                                                                                                  | IS         |
| [03\_4 Narratives Vocabulary exploration](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/03_4%20Narratives%20Vocabulary%20exploration.ipynb)                                                                                                                        | Narrative (statistical) vocabulary exploration                                                                                   | IS, HA     |
| [04\_1\_1 Anomaly feature Definition TRAIN set](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/04_1_1%20Anomaly%20feature%20Definition%20TRAIN%20set.ipynb)                                                                                                         | Target feature one-hot encoding (train set)                                                                                      | IS, HA, HH |
| [04\_1\_1\_2 Anomaly feature Exploration on TRAIN set](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/04_1_1_2%20Anomaly%20feature%20Exploration%20on%20TRAIN%20set.ipynb)                                                                                          | Study of the target variable (correlations etc.)                                                                                 | IS, HA     |
| [04\_1\_2 Anomaly feature Definition TEST set](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/04_1_2%20Anomaly%20feature%20Definition%20TEST%20set.ipynb.ipynb)                                                                                                     | Target feature one-hot encoding (final set)                                                                                      | IS, HA     |
| [05\_1 Anomaly Prediction - Baseline Model](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/05_1%20Anomaly%20Prediction%20-%20Baseline%20Model.ipynb)                                                                                                                | Modeling - Baseline model (Decision Tree) on train data set, SHAP analysis                                                                      | IS, HA     |
| [05\_2\_1 Anomaly Prediction - BOW Unsupervised feature selection OneVsAll](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/05_2_1%20Anomaly%20Prediction%20-%20BOW%20Unsupervised%20feature%20selection%20OneVsAll.ipynb)                                           | Modeling - Bag-of-Words with UNsupervised feature selection (DT and RF) on train data set                                        | HH         |
| [05\_2\_2 Anomaly Prediction - BOW Unsupervised feature selection Naive\_bayes](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/05_2_2%20Anomaly%20Prediction%20-%20BOW%20Unsupervised%20feature%20selection%20Naive_bayes.ipynb)                                    | Modeling - Bag-of-Words with UNsupervised feature selection (NB) on train data set                                               | HH         |
| [05\_3\_1 Anomaly Prediction - BOW Supervised feature selection (DT, RF, GB no grid)](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/05_3_1%20Anomaly%20Prediction%20-%20BOW%20Supervised%20feature%20selection%20(DT%2C%20RF%2C%20GB%20no%20grid).ipynb)           | Modeling - Bag-of-Words with supervised feature selection (DT, RF, GB without Hyperparameters optimisation) on train data set    | HA         |
| [05\_3\_2 Anomaly Prediction - BOW Supervised feature selection (DT, RF, GB Grid + SVM)](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/05_3_2%20Anomaly%20Prediction%20-%20BOW%20Supervised%20feature%20selection%20(DT%2C%20RF%2C%20GB%20Grid%20%2B%20SVM).ipynb) | Modeling - Bag-of-Words with supervised feature selection (DT, RF, GB with Hyperparameters optimisation + SVM) on train data set | HA         |
| [05\_4\_1 Anomaly Prediction - WordEmbedding 1vsAll](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/05_4_1%20Anomaly%20Prediction%20-%20WordEmbedding%201vsAll.ipynb)                                                                                               | Modeling - Word Embedding - 1vsAll mode on train data set                                                                        | IS         |
| [05\_4\_2 Anomaly Prediction - WordEmbedding PADDING EXPERIMENTS](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/05_4_2%20Anomaly%20Prediction%20-%20WordEmbedding%20PADDING%20EXPERIMENTS.ipynb)                                                                   | Modeling - Word Embedding parameter study on train data set                                                                      | IS         |
| [05\_5 Transformer model 7\_3\_9\_3\_UNfrozen](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/05_5%20Transformer%20model%207_3_9_3_UNfrozen.ipynb)                                                                                                                  | Modeling - best BERT-based transformer on train data set                                                                         | IS         |
| [06 Model Comparison Plot (Baseline DT\_Raw)](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/06%20Model%20Comparison%20Plot%20(Baseline%20DT_Raw).ipynb)                                                                                                            | Model comparison plotting metrics (f1-score, recall, precision, diff\_f1\_score…)                                                | IS, HA     |
| [07 BERT classes on GitHUB](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/07%20BERT%20classes%20on%20GitHUB.ipynb)                                                                                                                                                 | Modeling - transformer (best model) - refactored in classes : train on train data set & inference on final test set              | IS         |
| [08\_1 Plots\_final\_test\_set](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/08_1%20Plots_final_test_set.ipynb)                                                                                                                                                   | Plotting the results from the final test set                                                                                     | IS         |
| [09 Threshold optimization final test set](https://github.com/DataScientest-Studio/Aerobot/blob/main/notebooks/main/09%20Threshold%20optimization%20final%20test%20set.ipynb)                                                                                                                 | Threshold optimization algorithm                                                                                                 | IS         |

  
![Screenshot 2022-11-03 at 15 43 47](https://user-images.githubusercontent.com/97918270/199752422-9afd834e-edc8-499e-bb4e-acdde55431fb.png)
![Screenshot 2022-11-02 at 16 09 00](https://user-images.githubusercontent.com/97918270/199526484-0b62d731-8498-40b1-bd49-f268bc4249ec.png)
Flow diagram illustrating the mutual dependencies of the project's ```.ipynb```  notebooks and the flow of data. 
The large dashed arrows show the continuation of the path across the line break.
The box colors indicate the various types: 
- data files (white), model files (orange) 
- notebooks on data exploration (gray) notebooks generating outputs (blue)


## Data
The [data](./data) is composed of 3 subfolders:
- [ASRS database](./data/ASRS%20database) contains auxiliary files found on the ASRS website, e.g. ```CodingForm.pdf``` (label taxonomy) and important abbreviations.
- [models](./data/models)
- [transformed](./data/transformed)
These last 2 are practically empty, in order to avoid hosting large files on the GitHUB repo. They are populated with data once the user runs the program ```AeroBOT.py``` (see below).


## ```AeroBOT.py``` Program
For demonstration purposes, this program performs **inference** on the **final test set** using transformer model *11_3_3*. 
**Training** of this model takes at least an hour on Google Colab premises and thus exceeds the scope of a demo. 

```AeroBOT.py``` demonstrates the
- conversion of the data from a pandas.DataFrame to a HuggingFace :hugs: dataset 
- tokenization using the BERT tokenizer
- conversion of the data from HuggingFace :hugs: dataset to TensorFlow dataset
- inference using the saved model: calculation of the label probabilities ```y_pred_proba```, conversion of the probabilities to y_pred, i.e. binary (0,1) multilabels using ```threshold = 0.5```.

and prints a classification report.
It then saves .pkl files locally.

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

## Credits
- Ioannis STASINOPOULOS ([GitHub](https://github.com/Cochonaki)) / [LinkedIn](https://www.linkedin.com/in/ioannis-stasinopoulos/))
- Helene ASSIR ([GitHub](https://github.com/EleniAmorgos)) / [LinkedIn](https://www.linkedin.com/in/heleneassir/))

Project mentor:
Alban THUET (DataScientest) [LinkedIn](https://www.linkedin.com/in/alban-thuet-683365173/)
  
This project is licensed under the terms of the MIT license.
