import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path  

st.set_page_config(page_title="AeroBOT Demo",
                  page_icon="✈")
st.markdown("""
          # ⚙ Preprocessing
          ### Preparing narratives for Bag-of-words models
          """)

###############################################
# EXPORT THIS INTO A PACKAGE
# @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
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
####################################################################
def catplot_diff_allmodels(highlight_best_models=False):

  color_approach_model_list=['#6aa84f' ]+[ '#16a3e0' ]* 14 +['#0d5ddf'] + ['#962c61']*15 +[ '#766d6b'] *5+ ['#f14124']*5 
  color_ticks_approach_model_list=color_approach_model_list
  if highlight_best_models:
    color_approach_model_list=['#6aa84f' ]+[ 'w' ]* 13+['#16a3e0'] +['#0d5ddf'] + ['#962c61'] +[ 'w' ] *14 +[ 'w']+[ '#766d6b']+[ 'w'] *3+ ['#f14124']+[ 'w' ] *4 
    color_ticks_approach_model_list=['#6aa84f' ]+[ '#d3d3d3' ]* 13+['#16a3e0'] +['#0d5ddf'] + ['#962c61'] +[ '#d3d3d3' ] *14 +[ '#d3d3d3']+[ '#766d6b']+[ '#d3d3d3'] *3+ ['#f14124']+[ '#d3d3d3' ] *4 

  approach_model_palette=sns.color_palette(color_approach_model_list)

  boxplot = plt.figure()
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

  ## Les 2 lignes en LateX ci-dessous ne marchent pas ...
  #boxplot.set(xlabel=r'$\text{​​​​​​diff}​​​​​​_{​​​​​​\text{​​​​​​ f1-score}​​​​​​}​​​​​​^{​​​​​​\text{​​​​​​ model}​​​​​​}​​​​​​$ [s]')
  #boxplot.set(xlabel=r'$\text{​diff}​_{​\text{​ f1-score}​}​^{​\text{​ model}​}​$')

  # Vertical bar in black
  boxplot.axvline(0, ls='-',color='black',linewidth=2)

  return boxplot
###################################################################
def catplot_diff_bestmodels():
  # FOCUS ON BEST MODELS FOR EACH APPROACH
  color_approach_list=['#6aa84f',  '#16a3e0', '#0d5ddf', '#962c61', '#766d6b', '#f14124' ]
  approach_palette=sns.color_palette(color_approach_list)

  boxplot = plt.figure(figsize= (10, 3))
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
 
  # Vertical bar in black
  boxplot.axvline(0, ls='-',color='black',linewidth=2)

  return boxplot

####################################################################
# Definition of color coding for each model type (grey otherwise in function)
dict_approach_color={'(1) Base line model':'#6aa84f', 
            '(2) BoW Unsupervised feat. selection':'#16a3e0',
            '(3) BoW Supervised feat. selection':'#0d5ddf',
            '(4) Word-Embedding':'#962c61',
            '(5) BERT Unfrozen': '#f14124',
            '(5) BERT Frozen': '#766d6b',
            }

##############################################
# LOAD FILES
##############################################
# Define useful directories
streamlit_root_dir = Path.cwd()
AeroBOT_root_dir = streamlit_root_dir.parents[0]
trans_data_path = AeroBOT_root_dir.joinpath('data', 'transformed')

@st.cache
def load_df(filename):
  os.chdir(trans_data_path) 
  df = pd.read_csv(filename).drop(columns = ['Unnamed: 0'])
  os.chdir(streamlit_root_dir)
  print('Streamlit data loaded.')
  return df

model_results_diffBLM_bestmodel = load_df('model_results_diffBLM_bestmodel_20221207.csv')
base_line_vs_BERT_results = load_df('baseline_vs_best_BERT_20221207.csv')


st.header("👩‍✈️👨‍✈️ Expert's workspace")

st.markdown("""
          Welcome to your dashboard, dear ASRS experts!
          
          Here you can identify the model that performs best in classifying the narratives of your domain of expertise.
          """)
##############################################
# Interactiveness
##############################################
# Define choices for the user 
model_approaches = model_results_diffBLM_bestmodel['approach'].unique()

anomaly_tuple = (
    '01_Deviation / Discrepancy - Procedural',
    '02_Aircraft Equipment',
    '03_Conflict',
    '04_Inflight Event / Encounter',
    '05_ATC Issue',
    '06_Deviation - Altitude',
    '07_Deviation - Track / Heading',
    '08_Ground Event / Encounter',
    '09_Flight Deck / Cabin / Aircraft Event',
    '10_Ground Incursion',
    '11_Airspace Violation',
    '12_Deviation - Speed',
    '13_Ground Excursion',
    #'14_No Specific Anomaly Occurred'
    )
anomaly = 'Anomaly_' + st.selectbox(
    'Choose the anomaly label of interest from the drop-down list:', anomaly_tuple).split(sep = '_')[1]

approaches_to_plot = st.multiselect(
    'You can limit the plot to the desired model approach(es). Bag of Words (BoW) models are more interpretable, while BERT models are more perfomant. ',
    model_approaches,
    model_approaches)

highlight_best_models = False # global variable

# I THOUGHT THIS WOULD SPEED UP THE PLOTTING, BUT RUNNING create_anomaly_figs_dict() takes ages
# @st.cache
# def create_anomaly_figs_dict(anomaly_tuple, df):
#   anomaly_figs_dict = {anomaly: plot_diff_metric_universal(df_model_results=df.sort_values(by = ['import_order'], 
#                                                   ascending = False),
#                                                   anomaly_list=[anomaly],
#                                                   metric="f1-score",
#                                                   dict_model_color=dict_model_color)
#   for anomaly in anomaly_tuple}
#   return anomaly_figs_dict

# anomaly_figs_dict = create_anomaly_figs_dict(anomaly_tuple, model_results_diffBLM_bestmodel)

st.markdown("""
            The plot below shows 
            - either the absolute value of the _f1-score_ for the selected anomaly label
            - or the difference in _f1-score_ with respect to the baseline model (a Decision Tree - DT), see equation below. In this case positive and negative values illustrate the over- and underperformance of the model, respectively. 
            """)

st.latex(r'''
    \text{diff}_{\text{ f1-score}}^{\text{ model}} (\text{anomaly}) = \text{f1-score}^{\text{ model}} (\text{anomaly}) - \text{f1-score}^{\text{ baseline}} (\text{anomaly})
    ''')

abs_or_rel = st.radio(
    'Select how to plot the scores:',
    ('Absolute values', 'Difference with respect to baseline model (DT)'))
if abs_or_rel == 'Difference with respect to baseline model (DT)':
  modality_col = 'diff'
else:
  modality_col = 'absolute'

with st.spinner('Plotting...'):
  st.pyplot(plot_diff_metric_universal(df_model_results = model_results_diffBLM_bestmodel[model_results_diffBLM_bestmodel['approach'].isin(approaches_to_plot)].sort_values(by = ['import_order'], ascending = False),
                      modality_col = modality_col,
                      anomaly_list = [anomaly],
                      metric = "f1-score",
                      dict_model_color = dict_approach_color,
                      color_by='approach',
                      model_name='approach and model number').figure # see https://github.com/streamlit/streamlit/issues/2609
            )
  

  model_Nr = st.number_input('Insert a model number to see its details:',
                            value = 5,
                            min_value = model_results_diffBLM_bestmodel['import_order'].min(),
                            max_value = model_results_diffBLM_bestmodel['import_order'].max(),
                            step = 1)

  # df_details = model_results_diffBLM_bestmodel#.set_index('import_order')
  df_details = model_results_diffBLM_bestmodel.loc[(model_results_diffBLM_bestmodel['import_order'] == model_Nr) &
                          (model_results_diffBLM_bestmodel['anomaly'] == 'Anomaly_Conflict') &
                          (model_results_diffBLM_bestmodel['metric'] == 'f1-score'), ['import_order', 'model_label']]
  df_details.rename(columns = {'import_order': 'model#', 
                                'model_label': 'model details'}, 
                    inplace = True)
  # df_details = df_details.reindex(columns=['model_label'])
  # st.write(df_details.columns)
  st.write(df_details.loc[:, ['model#', 'model details']].reset_index(drop=True).set_index('model#'))

st.success('Page refresh successful.')