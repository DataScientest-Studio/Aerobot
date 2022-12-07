import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path  


st.set_page_config(page_title="AeroBOT Demo",
                  page_icon="✈") #🛩
st.markdown("""
          # 📊 Model Evaluation
          ### Choosing the best model
          """)

tab1, tab2 = st.tabs(["Cat", "Dog"])

with tab1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

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

  return barh

# def plot_diff_metric_universal(df_model_results, 
#                                modality_col,
#                                anomaly_list=[], 
#                                metric="f1-score", 
#                                dict_model_color={}
#                                ):
#     """
#     Plots, for each anomaly, the evolution of  "metric" 
#     Inputs: 
#     - model_results : a df containing the classification report metrics of our different "models" to plot
#       Models include : classifier type and modeling options such as  raw/PP narratives, std or under sampling, count_vectorizer options
#     - a list of anomaly features : if the list is empty : 
#     - metric : one of the model results metrics : "precision", "recall", "f1-score" or "support"
#     - dict_color : dictionnary defining a color for each type of model listed (grey if non listed)
    
#     Returns:
#     - 1 plot per anomaly listed
#       - for models using undersampling, the line  of the rectangle is thiner
#       - for models using raw narratives (vs PP), the line  of the rectangle is grey instead of black

#     """
#     if anomaly_list == [] :
#       anomaly_list = df_model_results['anomaly'].unique().tolist()
#     for anomaly in anomaly_list :
  
#       # Anomaly_label without the prefix "Anomaly_"
#       anomaly_label=anomaly.replace("Anomaly_", "")  
      
#       metric_row=metric
#       if modality_col == 'absolute': 
#         title_ToPlot=anomaly_label+" :\n" + metric
#       else:      
#         title_ToPlot=anomaly_label+" :\n  Difference of "+ metric+" vs. Baseline model "

#       # dataframe containing only the rows to plot
#       sub_df = df_model_results[(df_model_results['anomaly'] == anomaly) & (df_model_results['metric'] == metric_row)].copy()
#       # label of the model , including options
#       sub_df=sub_df.set_index('model_label')
#       # defining color, edgecolot, linewidth of the bar according to the model characteristics
#       sub_df['color']=sub_df['classifier'].apply(lambda x: dict_model_color[x] if x in list(dict_model_color.keys()) else 'grey')
#       sub_df['edgecolor']=sub_df['preprocessing'].apply(lambda x: 'grey' if x==0 else 'black')
#       sub_df['linewidth']=sub_df['undersampling'].apply(lambda x: 3 if x==0 else 1)
      
#       # Plot
#       fig = plt.figure()
#       plt.style.use('ggplot')
#       plt.rcParams['axes.titlesize'] = 15
#       plt.rcParams['axes.labelsize'] = 10
#       plt.rcParams['xtick.labelsize'] = 10
#       plt.rcParams['ytick.labelsize'] = 13
#       plt.rc('legend', fontsize=10)    # legend fontsize

      
#       num_classes = len(sub_df)
#       fig_shape=(8,num_classes//4)

#       colors=list(sub_df['color'])
#       edgecolors=list(sub_df['edgecolor'])
#       linewidths=list(sub_df['linewidth'])
#       iter_color = iter(colors)

#       sub_df[modality_col].plot.barh(title=title_ToPlot, 
#                                                       ylabel="Topics",
#                                                       color=colors,
#                                                       edgecolor=edgecolors,
#                                                       linewidth=linewidths,
#                                                       figsize=fig_shape) # indicative value for BERT models only: (8,7)
#       if metric != "support":
#         if modality_col == 'absolute': 
#           plt.xlim([0, 1])
#         else:
#           plt.xlim([-1,1])
#         plt.xticks([])
#         for i, v in enumerate(sub_df[modality_col]):
#           c = next(iter_color)
#           if v>=0 :
#             y=v
#           else :
#             y=v-0.2
#           plt.text(y, i,           # si bar au lieu de barh : inverser v et i
#                    " "+str(round(v*100,1))+"%", 
#                     color=c, 
#                     va='center', 
#                     fontweight='bold')

#       else : 
#           for i, v in enumerate(sub_df[modality_col]):
#             c = next(iter_color)
#             plt.text(v, i,           # si bar au lieu de barh : inverser v et i
#                     " "+str(int(v)), 
#                     color=c, 
#                     va='center', 
#                     fontweight='bold')

#     print(modality_col)
#     return fig
###################################################################
# Definition of color coding for each model type (grey otherwise in function)
dict_model_color={'Decision Tree':'#15B01A' , 
            'DecisionTreeClassifier':'#15B01A' , 
            'Decision Tree (Grid)':'#e69138' , 
            'Random Forest':'#008080' , 
            'RandomForestClassifier':'#008080' , 
            'Random Forest (Grid)':'#e69138' , 
            'naive bayes':'#674ea7' , 
            'Gradient Boosting':'#16a3e0' , 
            'Gradient Boosting (Grid)':'#e69138' , 
            'SVM':'#162d5a' ,
            'Word_Embedding':'#962c61',
          #  'BERT_BASE': '#f14124',
            'BERT_BASE UNFROZEN': '#f14124',
            'BERT_BASE FROZEN': '#766d6b',
            }

# Definition of color coding for each model type (grey otherwise in function)
dict_approach_color={'(1) Base line model':'#595959', 
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
    'Choose an anomaly label from the drop-down list:', anomaly_tuple).split(sep = '_')[1]

approaches_to_plot = st.multiselect(
    'You can limit the plot to the desired model approach(es)',
    model_approaches,
    model_approaches)

abs_or_rel = st.radio(
    'Select how to plot the scores:',
    ('Absolute values', 'Difference with respect to baseline model (DT)'))
if abs_or_rel == 'Difference with respect to baseline model (DT)':
  modality_col = 'diff'
else:
  modality_col = 'absolute'

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
            - either the absolute value of the _f1-score_ 
            - or the difference in _f1-score_ with respect to the baseline model, for the models in the selected model approaches :
            """)

st.latex(r'''
    \text{diff}_{\text{ f1-score}}^{\text{ model}} (\text{anomaly}) = \text{f1-score}^{\text{ model}} (\text{anomaly}) - \text{f1-score}^{\text{ baseline}} (\text{anomaly})
    ''')

# with st.spinner('Plotting...'):
#   st.pyplot(plot_diff_metric_universal(model_results_diffBLM_bestmodel[model_results_diffBLM_bestmodel['approach'].isin(approaches_to_plot)].sort_values(by = ['import_order'], ascending = False),
#                                                     modality_col,
#                                                     anomaly_list=[anomaly],
#                                                     metric="f1-score",
#                                                     dict_model_color=dict_model_color)
#                                                     )

with st.spinner('Plotting...'):
  st.pyplot(plot_diff_metric_universal(df_model_results = model_results_diffBLM_bestmodel[model_results_diffBLM_bestmodel['approach'].isin(approaches_to_plot)].sort_values(by = ['import_order'], ascending = False),
                      modality_col = modality_col,
                      anomaly_list = [anomaly],
                      metric = "f1-score",
                      dict_model_color = dict_approach_color,
                      color_by='approach',
                      model_name='approach and model number').figure # see https://github.com/streamlit/streamlit/issues/2609
            )

st.markdown("""
            Positive and negative values illustrate the over- and underperformance of the model, respectively. 
            """)
            
with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

st.markdown("""               
            We show for each model, a boxplot of the difference in f1-score with respect to the baseline model, for all 13 key anomalies
            """)
with st.spinner('Plotting...'):
  st.pyplot(
    sns.catplot(data=model_results_diffBLM_bestmodel[(model_results_diffBLM_bestmodel['metric']=='f1-score')&(model_results_diffBLM_bestmodel['anomaly']!='Anomaly_No Specific Anomaly Occurred')],
                y='model_label',
                x='1',
                kind ='box',
                orient='h',
                height=8)
          )

st.markdown("""
            Our criterion for polyvalence reads as follows: 
            """)

st.latex(r'''
    \begin{equation*}
    \begin{split}
    &\text{model is polyvalent}
    \\
    &\Leftrightarrow \text{diff}_{\text{ f1-score}}^{\text{ model}}(\text{anomaly}) > 0 \quad \forall \text{ anomaly}
    \\
    &\Leftrightarrow \text{model boxplot on the right of the 0 red vertical line}
    \end{split}
    \end{equation*} 
     ''')
    

st.markdown("""
            From all the models shown above, we summarize the best models of each modeling approach: 
            """)
# Catplot
# @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def catplot1():
  return     sns.catplot(data=model_results_diffBLM_bestmodel[(model_results_diffBLM_bestmodel['metric']=='f1-score') \
                                                    &(model_results_diffBLM_bestmodel['model_label'].isin(['Decision Tree/Raw/Std sampling/' \
                                                            , 'naive bayes_tfidfvectorizer_vocab_size:3000_PP'
                                                            ,'Best BoW Supervised feature selection Model'
                                                            ,'Word_Embedding/PP/Std sampling/'
                                                            ,'7_5_4_2_BERT_BASE_raw_FROZEN_concat_layers_NO_layer_11_flattened'
                                                            ,'7_3_9_3_BERT_BASE_raw_UNfrozen_layers_9_10_11_12_concat_layers_NO_last_hidden_state_CLS'])) \
                      
                                                    &(model_results_diffBLM_bestmodel['anomaly']!='14_No Specific Anomaly Occurred')] \
                ,y='model_label',x='1',kind ='box' ,orient='h',height=8)

with st.spinner('Plotting...'):
  st.pyplot(catplot1())

st.markdown("""
            ### Baseline model vs. best BERT model     
            """)          


# @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_baseline_vs_BERT(df, metric):
  fig = plt.figure(figsize = (15,10))
  palette = sns.color_palette(["#962c61", "#766d6b"]) #gray: 766d6b, "#16a3e0",

  df_for_barplot = df[(df['metric'] == metric)]
  b = sns.barplot(data = df_for_barplot, x = '1', y = 'anomaly', 
                  hue = 'model_label',
                  palette = palette)

  # Bert
  df_temp1 = df_for_barplot[df_for_barplot['model_label'] == 'best transformer']
  for i, v in zip(range(len(df_temp1['anomaly'])), df_temp1['1']):
    plt.text(v+0.005, i-.05,           
            str(np.round(100*v,1))+'%', 
            color="#962c61", 
            va='baseline', 
            fontweight='bold',
            fontsize = 13)
    
  # Baseline
  df_temp1 = df_for_barplot[df_for_barplot['model_label'] == 'baseline model (DT)']
  for i, v in zip(range(len(df_temp1['anomaly'])), df_temp1['1']):
    plt.text(v+0.005, i+.1,       
            str(np.round(100*v,1))+'%', 
            color="#766d6b", 
            va='top', 
            fontweight='bold',
            fontsize = 13)

  plt.rcParams['axes.titlesize'] = 23
  plt.rcParams['axes.labelsize'] = 23
  plt.rcParams['ytick.labelsize'] = 23
  plt.rc('legend', fontsize=20)    # legend fontsize

  b.legend_.set_title(None)
  plt.legend(loc='lower right')
  # b.legend().set_visible(False)
  plt.xlim([0,1])
  plt.xticks([])
  plt.xlabel(metric)
  plt.title(f'Baseline (DT) vs. best transformer model (BERT), metric: {metric}')

  return fig

with st.spinner('Plotting...'):
  st.pyplot(plot_baseline_vs_BERT(base_line_vs_BERT_results[base_line_vs_BERT_results['anomaly'] != '14_No Specific Anomaly Occurred'], 
            'f1-score'))

st.success('Page refreshed successfuly.')