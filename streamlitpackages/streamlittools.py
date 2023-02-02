import streamlit as st 
import os
from pathlib import Path
import base64
import inspect
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# project-specific functions and classes created in the 'Aerobot' project by Ioannis STASINOPOULOS

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url, size=70):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" height={size}px/>
        </a>'''
    return html_code

def get_image(img_name:str):
  img_path = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0].joinpath('ressources').joinpath(img_name)
  image = Image.open(img_path)
  return image


#########################################################################################################
# For Modeling page
#########################################################################################################
fig_bg_color = "#e8e3d4"

streamlit_root_dir = Path.cwd()
AeroBOT_root_dir = streamlit_root_dir#.parents[0] # comment out if streamlit main file in the GitHUB root 
trans_data_path = AeroBOT_root_dir.joinpath('data', 'transformed')

@st.cache
def load_df(filename):
  os.chdir(trans_data_path) 
  df = pd.read_csv(filename).drop(columns = ['Unnamed: 0'])
  os.chdir(streamlit_root_dir)
  print('Streamlit data loaded.')
  return df
model_results_diffBLM_bestmodel = load_df('model_results_diffBLM_bestmodel_20221207.csv')

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_diff_metric_universal(df_model_results, 
                                modality_col,anomaly_list=[],
                                metric="f1-score",
                                dict_model_color={},
                                color_by='model',
                                model_name='model_label'):
  """
  Plots, for each anomaly, the evolution of  "metric" 
  Inputs: 
  - model_results : a df containing the classification report metrics of our different "models" to plot
    Models include : classifier type and modeling options such as  raw/PP narratives, std or under sampling, count_vectorizer options
  - a list of anomaly features : if the list is empty : 
  - metric : one of the model results metrics : "precision", "recall", "f1-score" or "support"
  - dict_color : dictionnary defining a color for each type of model listed (grey if non listed)
    - color_by = 'model' (default or 'approach'
  - model_name = 'model_label'(default) or any columns with a model name   
  Returns:
  - 1 plot per anomaly listed
    - for models using undersampling, the line  of the rectangle is thiner
    - for models using raw narratives (vs PP), the line  of the rectangle is grey instead of black

  """
  if anomaly_list==[] :
    anomaly_list=df_model_results['anomaly'].unique().tolist()
  for anomaly in anomaly_list :

    # Anomaly_label without the prefix "Anomaly_"
    anomaly_label=anomaly.replace("Anomaly_", "")  
    
    metric_row=metric
    if modality_col == 'absolute': 
      title_ToPlot=anomaly_label+" :\n" + metric

    else:      
      title_ToPlot=anomaly_label+" :\n  Difference of "+ metric+" vs. Baseline model "

    # dataframe containing only the rows to plot
    sub_df = df_model_results[(df_model_results['anomaly'] == anomaly) & (df_model_results['metric'] == metric_row)] .copy()
    # label of the model , including options
    sub_df=sub_df.set_index(model_name)
    # defining color, edgecolot, linewidth of the bar according to the model characteristics
    if color_by=='model' :
      sub_df['color']=sub_df['classifier'].apply(lambda x: dict_model_color[x] if x in list(dict_model_color.keys()) else 'grey')
      sub_df['edgecolor']=sub_df['preprocessing'].apply(lambda x: 'grey' if x==0 else 'black')
      sub_df['linewidth']=sub_df['undersampling'].apply(lambda x: 3 if x==0 else 1)
    elif color_by=='approach' :
      sub_df['color']=sub_df['approach'].apply(lambda x: dict_model_color[x] if x in list(dict_model_color.keys()) else 'grey')
      sub_df['edgecolor']='black'
      sub_df['linewidth']= 1

    # Plot
    fig = plt.figure()
    fig.set_facecolor(fig_bg_color) 
    plt.style.use('ggplot')
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 13
    plt.rc('legend', fontsize=10)    # legend fontsize

    num_classes = len(sub_df)
    fig_shape=(8,num_classes//3.5)

    colors=list(sub_df['color'])
    edgecolors=list(sub_df['edgecolor'])
    linewidths=list(sub_df['linewidth'])
    iter_color = iter(colors)

    barh=sub_df[modality_col].plot.barh(title=title_ToPlot, 
                                                    ylabel="Topics",
                                                    color=colors,
                                                    edgecolor=edgecolors,
                                                    linewidth=linewidths,
                                                    figsize=fig_shape) # indicative value for BERT models only: (8,7)

    # ytick labels in color : according to approach + highlight_best_models
    for ytick, color in zip(barh.get_yticklabels(), colors):
      ytick.set_color(color)
    
    if metric != "support":
      if modality_col == 'absolute': 
        plt.xlim([0, 1])
      else:
        plt.xlim([-1,1])
      plt.xticks([])
      for i, v in enumerate(sub_df[modality_col]):
        c = next(iter_color)
        if v>=0 :
          y=v
        else :
          y=v-0.2
        plt.text(y, i,           # si bar au lieu de barh : inverser v et i
                  " "+str(round(v*100,1))+"%", 
                  color=c, 
                  va='center', 
                  fontweight='bold')

    else : 
        for i, v in enumerate(sub_df[modality_col]):
          c = next(iter_color)
          plt.text(v, i,           # si bar au lieu de barh : inverser v et i
                  " "+str(int(v)), 
                  color=c, 
                  va='center', 
                  fontweight='bold')
    plt.ylabel("Approach and Model number", fontsize = 14)
    
    if modality_col == 'absolute': 
      barh.set_xlabel('f1-score', fontsize = 16)
    
    else: 
      barh.set_xlabel(r'$diff^{ model}_{ f1-score} (anomaly) $', fontsize = 16)


  return barh

def catplot_diff_allmodels(highlight_best_models=False):

  color_approach_model_list=['#6aa84f' ]+[ '#16a3e0' ]* 14 +['#0d5ddf'] + ['#962c61']*15 +[ '#766d6b'] *5+ ['#f14124']*5 
  color_ticks_approach_model_list=color_approach_model_list
  if highlight_best_models:
    color_approach_model_list=['#6aa84f' ]+[ 'w' ]* 13+['#16a3e0'] +['#0d5ddf'] + ['#962c61'] +[ 'w' ] *14 +[ 'w']+[ '#766d6b']+[ 'w'] *3+ ['#f14124']+[ 'w' ] *4 
    color_ticks_approach_model_list=['#6aa84f' ]+[ '#d3d3d3' ]* 13+['#16a3e0'] +['#0d5ddf'] + ['#962c61'] +[ '#d3d3d3' ] *14 +[ '#d3d3d3']+[ '#766d6b']+[ '#d3d3d3'] *3+ ['#f14124']+[ '#d3d3d3' ] *4 

  approach_model_palette=sns.color_palette(color_approach_model_list)

  boxplot = plt.figure()
  boxplot.set_facecolor(fig_bg_color)
  ax = plt.axes()
  ax.set_facecolor("orange") 
  boxplot = plt.figure(figsize= (10, 12))
  # sns.set(rc={'figure.figsize':(10,12)})
  sns.set(font_scale = 1.1)
  boxplot=sns.boxplot(data=model_results_diffBLM_bestmodel[(model_results_diffBLM_bestmodel['metric']=='f1-score')&(model_results_diffBLM_bestmodel['anomaly']!='Anomaly_No Specific Anomaly Occurred')],
              y='approach and model number',
              x='diff',
              #kind ='box',
              orient='h',
            #  height=8, 
              palette=approach_model_palette)
  boxplot.set_facecolor(fig_bg_color) 

  # ytick labels in color : according to approach + highlight_best_models
  for ytick, color in zip(boxplot.get_yticklabels(), color_ticks_approach_model_list):
    ytick.set_color(color)

  # Limits min/max of x-axis
  boxplot.set(xlim=(-0.8, 0.4))

  # Titles
  #boxplot.set_xlabel("diff_f1_score", fontsize = 16)
  # boxplot.set_xlabel('$\t{​diff}​_{​\t{​ f1-score}​}​^{​\t{​ model}​}​$', fontsize = 16)
  boxplot.set_xlabel(r'$diff^{ model}_{ f1-score}$ (all anomalies)', fontsize = 16)
  boxplot.set_ylabel("Approach and Model number", fontsize = 16)
  boxplot.set_title("Difference of f1-score vs Baseline model", fontsize = 18, weight='bold')
  boxplot.set_facecolor(fig_bg_color)
  ## Les 2 lignes en LateX ci-dessous ne marchent pas ...
  #boxplot.set(xlabel=r'$\text{​​​​​​diff}​​​​​​_{​​​​​​\text{​​​​​​ f1-score}​​​​​​}​​​​​​^{​​​​​​\text{​​​​​​ model}​​​​​​}​​​​​​$ [s]')
  #boxplot.set(xlabel=r'$\text{​diff}​_{​\text{​ f1-score}​}​^{​\text{​ model}​}​$')

  # Vertical bar in black
  boxplot.axvline(0, ls='-',color='black',linewidth=2)

  return boxplot

def catplot_diff_bestmodels():
  # FOCUS ON BEST MODELS FOR EACH APPROACH
  color_approach_list=['#6aa84f',  '#16a3e0', '#0d5ddf', '#962c61', '#766d6b', '#f14124' ]
  approach_palette=sns.color_palette(color_approach_list)

  boxplot = plt.figure(figsize= (10, 3))
  boxplot.set_facecolor(fig_bg_color) 
  # sns.set(rc={'figure.figsize':(10,3)})
  sns.set(font_scale = 1.1)
  boxplot=sns.boxplot(data=model_results_diffBLM_bestmodel[(model_results_diffBLM_bestmodel['metric']=='f1-score') \
                                                  &(model_results_diffBLM_bestmodel['Best approach model']==True) \
                                                  &(model_results_diffBLM_bestmodel['anomaly']!='Anomaly_No Specific Anomaly Occurred')], \
                      y='approach and model number',
                      x='diff' ,
                      orient='h', 
                      palette=approach_palette)
  
  # ytick labels in color : according to approach + highlight_best_models
  for ytick, color in zip(boxplot.get_yticklabels(), color_approach_list):
      ytick.set_color(color)

  # Limits min/max of x-axis
  boxplot.set(xlim=(-0.8, 0.4))
  
  # Titles
  #boxplot.set_xlabel("diff_f1_score", fontsize = 14)
  boxplot.set_xlabel(r'$diff^{ model}_{ f1-score}$ (all anomalies)', fontsize = 16)
  boxplot.set_ylabel("Approach and Model number", fontsize = 14)
  boxplot.set_title("Difference of f1-score vs Baseline model \n Best model by approach", fontsize = 18, weight='bold')
  boxplot.set_facecolor(fig_bg_color)
  # Vertical bar in black
  boxplot.axvline(0, ls='-',color='black',linewidth=2)

  return boxplot

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_baseline_vs_BERT(df, metric):
    fig = plt.figure(figsize = (15,10))
    fig.set_facecolor(fig_bg_color) 
    palette = sns.color_palette(["#f14124", "#6aa84f"]) #gray: #595959, "#16a3e0" green: #6aa84f

    df_for_barplot = df[(df['metric'] == metric)]
    b = sns.barplot(data = df_for_barplot, x = '1', y = 'anomaly', 
                    hue = 'model_label',
                    palette = palette)
    b.set_facecolor(fig_bg_color)

    # Bert
    df_temp1 = df_for_barplot[df_for_barplot['model_label'] == 'best transformer']
    for i, v in zip(range(len(df_temp1['anomaly'])), df_temp1['1']):
        plt.text(v+0.005, i-.05,           
                str(np.round(100*v,1))+'%', 
                color="#f14124", 
                va='baseline', 
                fontweight='bold',
                fontsize = 13)
        
    # Baseline
    df_temp1 = df_for_barplot[df_for_barplot['model_label'] == 'baseline model (DT)']
    for i, v in zip(range(len(df_temp1['anomaly'])), df_temp1['1']):
        plt.text(v+0.005, i+.1,       
                str(np.round(100*v,1))+'%', 
                color="#6aa84f", 
                va='top', 
                fontweight='bold',
                fontsize = 13)

    plt.rcParams['axes.titlesize'] = 23
    plt.rcParams['axes.labelsize'] = 23
    plt.rcParams['ytick.labelsize'] = 23
    plt.rc('legend', fontsize=20)    # legend fontsize

    b.legend_.set_title(None)
    b.set_facecolor(fig_bg_color)
    plt.legend(loc='lower right')
    # b.legend().set_visible(False)
    plt.xlim([0,1])
    plt.xticks([])
    plt.xlabel(metric)
    plt.title(f'Baseline (DT) vs. best transformer model (BERT #93), metric: {metric}')

    return fig
#########################################################################################################