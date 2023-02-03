import streamlit as st
import os
from pathlib import Path
import inspect
from streamlitpackages import *

# Configure sidebar
streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0])
with st.sidebar:
  st.header("Contact")
  logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/ioannis-stasinopoulos/', 20)
  st.write(f'''<a href="https://www.linkedin.com/in/ioannis-stasinopoulos/" style="text-decoration: none;color:black">Ioannis STASINOPOULOS</a> {logo_linkedin}''', unsafe_allow_html=True) 
  st.write(f'''<a href="https://www.linkedin.com/in/heleneassir/" style="text-decoration: none;color:black">Hélène ASSIR</a> {logo_linkedin}''', unsafe_allow_html=True) 


st.markdown("""
          # 📊 Model Evaluation
          ### Choosing the best model for your work
          """)

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
AeroBOT_root_dir = streamlit_root_dir #.parents[0] # comment out if streamlit main file in the GitHUB root 
trans_data_path = AeroBOT_root_dir.joinpath('data', 'transformed')

base_line_vs_BERT_results = load_df('baseline_vs_best_BERT_20221207.csv')

tab1, tab2 = st.tabs(["👩‍✈️👨‍✈️ Workspace for ASRS Experts", 
                    "👩‍💼👨‍💼 Workspace for ASRS Admins"])

with tab1:
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

      df_details = model_results_diffBLM_bestmodel.loc[(model_results_diffBLM_bestmodel['import_order'] == model_Nr) &
                              (model_results_diffBLM_bestmodel['anomaly'] == 'Anomaly_Conflict') &
                              (model_results_diffBLM_bestmodel['metric'] == 'f1-score'), ['import_order', 'model_label']]
      df_details.rename(columns = {'import_order': 'model#', 
                                   'model_label': 'model details'}, 
                        inplace = True)

      st.write(df_details.loc[:, ['model#', 'model details']].reset_index(drop=True).set_index('model#'))

with tab2:
  st.header("👩‍💼👨‍💼 ASRS admins' workspace")

  st.markdown("""
          Welcome to your dashboard, dear ASRS administrator!
          
          _Dont' know in which technology to invest for improving the narrative classification process at NASA ?_
          
          Here you can identify which models are _versatile_ and which perform best, taking into account **all anomaly labels**.
          """)
  st.markdown("""
                Our criterion for polyvalence reads as follows: 
                """)

  st.latex(r'''
      \begin{equation*}
      \begin{split}
      &\text{model is } versatile
      \\
      &\Leftrightarrow \text{diff}_{\text{ f1-score}}^{\text{ model}}(\text{anomaly}) > 0 \quad \forall \text{ anomaly}
      \\
      &\Leftrightarrow \text{model boxplot on the right of the black vertical line}
      \end{split}
      \end{equation*} 
      ''')

  st.markdown("""               
              We show for each model, a boxplot of the difference in f1-score with respect to the baseline model, *for all 13 anomalies*
              
              For a given approach, the 'best' model is defined as the one with the highest median. 
              """)

  highlight_best_models = st.checkbox('Highlight the best model of each approach')

  with st.spinner('Plotting...'):
    st.pyplot(
      catplot_diff_allmodels(highlight_best_models).figure
            )

  st.markdown("""
              From all the models shown above, we group together the best models of each modeling approach: 
              """)

  with st.spinner('Plotting...'):
    st.pyplot(
              # catplot1()
              catplot_diff_bestmodels().figure
              )

  st.markdown("""
              We see that models the best models from the approaches
              - (3) BoW Supervised feat. selection 
              - (4) Word-Embedding
              - (5) BERT Unfrozen
              are versatile. 
              
              The latter, with the highest median, yields the best performance.
            """)


  st.markdown("""
              ### Interpretability vs. performance
              Below, we compare the _f1-score_ as obtained from the Baseline (DT) vs. from the best transformer model (BERT #93), 
              for all anomaly labels. 
              """)          

  with st.spinner('Plotting...'):
    st.pyplot(plot_baseline_vs_BERT(base_line_vs_BERT_results[base_line_vs_BERT_results['anomaly'] != '14_No Specific Anomaly Occurred'], 
              'f1-score'))