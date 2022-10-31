# Aerobot

## Explications et Instructions

<a href="url"><img src="https://user-images.githubusercontent.com/97918270/167825700-7ed773a3-8088-4adb-9c81-3bed1f3a10a4.png" align="left" height="248" ></a>

Ce repository contient les fichiers nécessaires à l'initialisation d'un projet fil-rouge dans le cadre de votre formation [DataScientest](https://datascientest.com/).

Il contient principalement le présent fichier README.md et un template d'application [Streamlit](https://streamlit.io/).

**README**

Le fichier README.md est un élément central de tout repository git. Il permet de présenter votre projet, ses objectifs, ainsi que d'expliquer comment installer et lancer le projet, ou même y contribuer.

Vous devrez donc modifier différentes sections du présent README.md, afin d'y inclure les informations nécessaires.

- Complétez **en anglais** les sections (`## Presentation` et `## Installation` `## Streamlit App`) en suivant les instructions présentes dans ces sections.
- Supprimer la présente section (`## Explications et Instructions`)

**Application Streamlit**

Un template d'application [Streamlit](https://streamlit.io/) est disponible dans le dossier [`streamlit_app`](streamlit_app). Vous pouvez partir de ce template pour mettre en avant votre projet.

## Presentation

Complétez cette section **en anglais** avec une brève description de votre projet, le contexte (en incluant un lien vers le parcours DataScientest), et les objectifs.

Vous pouvez également ajouter une brève présentation des membres de l'équipe avec des liens vers vos réseaux respectifs (GitHub et/ou LinkedIn par exemple).

**Exemple :**

This repository contains the code for our project **PROJECT_NAME**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to **...**

This project was developed by the following team :

- Ioannis STASINOPOULOS ([GitHub](https://github.com/Cochonaki)) / [LinkedIn](https://www.linkedin.com/in/ioannis-stasinopoulos/))

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) with:

```
pip install -r requirements.txt
```
___Note:___ Use the ```requirements_for_Google_Colab.txt``` on Google Colab. 

This file was generated with 
```
! pip freeze > requirements.txt
```
on Google Colab and contains an exhaustive list of the necessary packages.

### Installation in a (mini)conda environment
Pre-requisites: 
- (mini)conda is installed.
- Aerbot repo is cloned locally.

1. Create a conda environment named e.g. 'aerobot_venv'
```
conda create --name aerobot_venv python
```
This installs python and basic packages, e.g. pip, necessary to install the project packages later on.

2. Activate the environment
```
conda activate aerobot_local
```

3. Navigate to the local mirror of the Aerobot repo using ```cd``` 
and nstall the AeroBOT packages into your conda environment:
```
pip install -r requirements.txt
```
This takes several minutes.

4. Check the installation by listing the installed packages in 'aerobot_venv'.
```
conda list 
```
5. Execute the ```AeroBOT.py``` program
(make sure you have cd to the location of ```AeroBOT.py```)
```
python AeroBOT.py 10
```
This will infer the transformer model on the first 10 entries of the final test set.
The program informs the user on its actions by printing info in the command line.
If necessary, run
```
python AeroBOT.py --help
```
for help.

## Streamlit App

**Add explanations on how to use the app.**

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
