import streamlit as st
import os
from pathlib import Path
import inspect
from streamlitpackages import get_img_with_href, get_image

# st.set_page_config(page_title="AeroBOT Demo",
#                   page_icon="✈")

# Configure sidebar
streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0])
with st.sidebar:
  st.header("Contact")
  logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/ioannis-stasinopoulos/', 20)
  st.write(f'''<a href="https://www.linkedin.com/in/ioannis-stasinopoulos/" style="text-decoration: none;color:black">Ioannis STASINOPOULOS</a> {logo_linkedin}''', unsafe_allow_html=True) 
  st.write(f'''<a href="https://www.linkedin.com/in/heleneassir/" style="text-decoration: none;color:black">Hélène ASSIR</a> {logo_linkedin}''', unsafe_allow_html=True) 

st.markdown("""# 💾 Data""")


tab1, tab2, tab3 = st.tabs(["ASRS Database", "Narratives", 'Target Feature'])

with tab1:

  st.markdown("""### NASA’s Aviation Safety Reporting System
  Citing NASA’s website: The open-source [ASRS database](https://asrs.arc.nasa.gov/search/database.html) is the world's largest repository of voluntary, confidential safety information provided by aviation's frontline personnel, including pilots, controllers, mechanics, flight attendants, and dispatchers. 
  Incident reports are read and analyzed by ASRS's corps of aviation safety analysts: experienced pilots, air traffic controllers, and mechanics. 
  Each report received by the ASRS is read by a minimum of two analysts.

  ASRS's database addresses a variety of aviation safety issues and includes the narratives submitted by reporters (after they have been sanitized for identifying details). The database also contains coded information by expert analysts from the original report which is used for data retrieval and analyses. 

  The associated [Search Engine](https://akama.arc.nasa.gov/ASRSDBOnline/QueryWizard_Filter.aspx#) allows to select data on a wide range of criteria and obtain subsets, e.g. 

  - Electrical failures for General Aviation aircraft.
  - Flight Attendant reported incidents involving passenger misconduct where alcohol was a factor, and there was an assault or physical aggression.
  - Avionics problems that may result from the influence of passenger electronic devices.
  - see other thematic [report sets](https://asrs.arc.nasa.gov/search/reportsets.html)

  """)
  st.image(get_image(img_name = 'ASRS_overview.png'), 
          caption="Snapshot of the ASRS Online search engine. The 'Text:Narrative' and ‘Event Type-Anomaly’ are the feature and target variable in AeroBOT project, respectively.")

  st.markdown("""### Data perimeter in AeroBOT
  The figure below shows that the data in the ASRS database are mainly from the USA; there is a negligible amount of Canadian data.
  We also observe a high ratio of geographically anonymized data and we do not observe a monotonous trend over time. 

  Possible interpretations include: 
  - evolving report filtering procedure / criteria 
  - according to ASRS’ statement: “the COVID-19 Pandemic caused a decrease in intake. Intake in 2020 averaged about 1,239 reports per week or 5,471 reports per month.”
  """)
  st.image(get_image(img_name = 'Data_per_year.png'), 
          caption='Time evolution and geographic distribution of data, after reviewing by the ASRS.')

with tab2:
  st.markdown("""### Narratives
  #### Visual exploration
  We show a wordcloud of all narratives inside the train set, after english stop-word filtering and removal of the following punctuation: '.', ';', '[', ']'. 
    
  The plot is generated with [wordcloud.WordCloud](https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html), which returns unigrams and bigrams by default. 
   Therefore, the bigrams 'Flight Attendant' and 'First Officer' are included in the figure below.""")
  st.image(get_image(img_name = 'wordcloud.png'), 
          caption='Wordcloud of all narratives in the train set after english stop-word filtering.')

  st.markdown("""
  We summarize our observations on the narratives available on the ASRS database:

  - Reports are written in formal english, with correct spelling.

  - They contain expressions / N-grams, e.g. ‘Air carrier’, ‘Air Traffic Area’. Some of these expressions are sometimes abbreviated, e.g. ‘ATA’ for ‘Air Traffic Area’

  - They contain technical terms specific to aeronautics. It requires expert knowledge to understand the terms, e.g. the bigramm HOLD SHORT instructs an aircraft to hold at least 200 ft from the edge of the runway while awaiting permission to cross or proceed onto the runway. 

  - They contain abbreviations, some of which are found in ‘normal’ English (HQ-Headquarters, AM/PM for time indications, ASAP, NASA), others are specific to aeronautics (TCAS-Traffic collision avoidance system, PACOT-Pacific Organised Track System, ARTCC-Air Route Traffic Control Centers). 

  - A [list](https://asrs.arc.nasa.gov/docs/dbol/ASRS_Abbreviations.pdf) of ~460 abbreviations is provided on the ASRS website. We found that this list is not exhaustive, i.e. there are abbreviations in the narratives that are not listed.

  - Contractions of words are found: ‘FLT’ for ‘flight’, ‘PLT’ for ‘pilot’, ‘APT’ for ‘airport’, ... The word ‘Aircraft’ is found as such inside 43.000 narratives and as ‘ACFT’ in 26.000 other narratives

  - The presence of words like ‘however’, ‘instead’, ‘even though’, ‘still’, ‘thought’, ‘believe’, ‘apparently’, ‘told us’ indicate the presence of conflicting situations, indirect speech and syntactic complexity. 

  - Quantitative (numeric) information is present, e.g. altitude levels (FL200 denotes  a Flight Level of 20,000 feet), headings (7R means right runway, oriented at 070 degrees). Numeric separators vary (; vs ,).

  - ASRS reports are anonymised (‘sanitized’ for not identifying details): some elements of the narratives are summarized in brackets or anonymized as: ‘ZZZ’, e.g. the aircraft type B747 might be substituted by simply ‘X’ or ‘Y’. This preserves the anonymity of the reporter, aircraft carrier / owner etc. and unavoidably causes a loss of information for some reports compared to the original ones. It is beyond our power to recover that lost information. 

  """)
  st.markdown("""#### Statistical exploration 
  We now show  a statistical exploration of our narratives.

  Using the *sklearn.feature_extraction.text.CountVectorizer*, we count the number of tokens in each narrative, after tokenization and stemming.
  In the following figure, we plot the CountVectorizer’s vocabulary length, as a function of the max_df and min_df setting passed to the CountVectorizer.

  These settings mean the following: 

  - The vocabulary ignores terms that have a document frequency strictly higher than 'max_df'.
  - The vocabulary ignores terms that have a document frequency strictly lower than 'min_df'.
  """)
  st.image(get_image(img_name = 'Zipf_curve.png'), 
          width = 350,
          caption='Vocabulary length (in thousands) as a function of the max_df (cyan) and min_df (dark red) setting of the CountVectorizer, respectively. Calculated values are represented by points on the curves; the curves are an interpolation between the points.')

  st.markdown("""
  We observe in the figure that both curves show a convergence for high values of the parameter. Their trends are inverted: the blue curve increases with increasing max_df value, while the dark red curve decreases. 
  This means that only the ‘min_df’ setting helps to reduce the vocabulary length and by extension, the calculation effort for our algorithms. 

  For example: at min_df = 1200 we read in the curve that ~1,200 words appear in more than 1,200 narratives (the two numbers coincide by chance). 
  E.g. the occurrence of 'aircraft’ can reach up to approx. 40,000 narratives, i.e. 50% of the corpus size (without even considering that there are many more occurrences of 'aircraft' in the form of the contraction 'ACFT')!

  Looking at the alphabetical list of the above-mentioned 1,200 words (not shown here), we observe that: 

  - it contains mostly meaningful words (time, aircraft, would, land, call, us, turn, back, fligh, ask) and that we find only a few 'strange' tokens such as 'zzz' (this stems from ASRS’ anonymization process)

  - only 48 tokens begin with a number, all of them probably being pure numbers in the text (headings and altitudes mostly)

  This means that increasing the min_df setting implies ending up with only the most ‘common’ words in our corpus, such as ‘airplane’ and loosing terms that are made of numeric values, e.g. altitude or runway orientation indications, e.g. 12L, which means left runway with 120° orientation, a wide-spread convention in the aeronautical world. The CountVectorizer considers these expressions as ‘rare tokens’, only because the particular numeric value is rare, i.e. there may be many runway indications in the corpus, but each one with a slightly different number.

  Looking at the problem from another side confirms this picture: when we set max_df = 6, we obtain ~40,300 words that appear in less than 6 narratives! 

  The list contains: 

  - surprising words, e.g. 'tetanus', 'tetrahedron’, 'teadrop’, 'syrup’, 'sycamor' (a wood species), 'swimmer’, ’swine’, 'sudoku'

  - words/codes, some with atypical letter repetition: 'zinta', 'w041’, ’zz0’, 'zztop’, 'zzzsecond’, ‘zzzzzzzz’, 'xxxxxxxxxxx', yxyx', 'yy3', 'yahoo’, 'xz47z'. Most of them are the result of ASRS’ anonymization procedure: 'zz0' may denote e.g. an anonymized aircraft in contrast to another anonymized aircraft 'zz1' in the same narrative.

  - hundreds of tokens that are numbers or contain numbers; we give some examples, along with our interpretation: 

    - '00237redeye'
    - '01014g22kt' wind direction: 10deg., speed: 10kts, gusts = 22kt (?)
    - '06062816000’
    - '075ft' altitude = 7500 ft.
    - '10400ft' altitude
    - '0900' time?
    - '10am' definitely time
    - '1000lbs' weight
    - '100hp' horsepower
    - '100knot' aircraft speed
    - '1013hpa' sea-level pressure indication in hPa
    - '101r' runway indication
    - '07left' runway indication 
    - '10deg' angle indication
    - '10mile' distance

  In order to deal with this, we process our narrative data by grouping together such numerical values under a common generic term, i.e. we substitute 12L by <RUNWAY>. This will increase the document frequency of <RUNWAY> and potentially pass the min_df threshold used. 

  To achieve this substitution as most efficiently as possible, we use regular expressions (Regex).
  """)

st.markdown(""" #### Case style clustering
Some narratives are entiretly written in uppercase letters and others contain some uppercase words. 
This raises questions as to how process abbreviations appropriately: 
in uppercase narratives, abbreviations in uppercase letters such as SO (second officer) and IF (intermediate fix) may be mistaken with the common words “so” and “if”. 

In AeroBOT, we qualify the case style of each narrative by defining specific features: 
in each sentence, we label the staring word and the following ones into one of the three following categories:
- lower case
- UPPERCASE 
- Capital letters

Then, in each narrative, we count how many sentence starting words belong to each of the three categories listed above. 
We do the the same for the following words in each sentece. 
This gives us 6 features that describe the case style of a narrative.

Applying a K-means clustering with the elbow method, we cluster the narratives into three clusters:
- “lower case”: most of the starting word of sentences start with a capital letter. The following words are in lower case.
- “upper case”: almost all words written exclusively in UPPER CASE.
- "mix case”: about 80% of the starting words start with a capital letter, and 20% are fully UPPER CASE.
""")
st.image(get_image(img_name = 'case_clusters_with_examples.png'), 
        caption='Top: overview of the three clusters that resulted from our analysis. Bottom: examples for each cluster. \
        Notice in the "mix case" example that only one sentence starts with an upper case word (an acronym): "VNAV made all..." \
        and that there are several acronyms written in upper case.')

st.markdown("""
_What is the distribution of the narratives among this clusters?_

The pie chart below shows that about 39% of the narratives are written in upper case.
""")
st.image(get_image(img_name = 'pie_chart.png'), 
        caption='Distribution of the 3 case style clusters in the corpus.')        

st.markdown("""
_What is the distribution of the narratives among this clusters **over the years**?_
""")
st.image(get_image(img_name = 'cluster_distr.png'), 
        caption='Distribution of the 3 case style clusters over the years.')

st.markdown("""
Until 2008, narratives were written in uppercase letters. The fully upper case style was not used at all after 2008. 
""")

st.markdown(""" #### Usage of abbreviations (TLP)
In this section, we explain our motivation for replacing abbreviations in our TLP Preprocessing.

Dealing with the technical language was one of the key aspects of our project. In our case, approximately 460 abbreviations are listed in the ASRS abbreviation list, such as:
- HQ (headquarters), AM/PM , ASAP, NASA, etc. which are known in the usual language,
- but also aeronautical terms, e.g. ATC (Air Traffic Control), AFDS (Autopilot and Flight Director System), AFSS (Automated Flight Service Station).

For instance,  “Aircraft” is referred to as ‘aircraft’ in 43,000 narratives and as ‘ACFT’ in 26,000 other narratives. 
In a Bag of Words approach, our aim is to correctly count technical words and expressions, avoiding duplicates.

The following figures show the usage of abbreviations in the two following cases:

- Contraction of words: They represent one third of the abbreviation count (~160).  Example: ‘ACFT’ for ‘aircraft’. The document frequencies of abbreviated form vs. full-text form show a rather mixed use of both forms

- Expressions’ acronyms : They represent two thirds of the abbreviation count (~300). Example: ‘ATC’  for  ‘air traffic control’. The document frequencies of abbreviated form vs. full-text form show that one form is more dominant: 'asap' and 'Flight attendant' dominate over their respective counter parts 'as soon as possible' and 'FA'. Exceptions to this are e.g. FO/First officer, PF/Pilot flying; here we find both with approximately equal occurence.
""")

st.image(get_image(img_name = 'use_of_contractions.png'), 
        caption='Usage of contractions in the narratives.')

st.image(get_image(img_name = 'use_of_acronyms.png'), 
        caption='Usage of expressions’ acronyms in the narratives.')




with tab3:
  st.markdown("""### Target feature: Anomaly
  We show the distribution of ‘Anomaly’ labels in the training data. The imbalance in the distribution of the labels is striking: the most frequent label (‘Deviation / Discrepancy - Procedural’) has ~30x higher occurrence compared to the rarest one (‘Ground Excursion’).
  """)
  st.image(get_image(img_name = 'cntplot_anomalies.png'), 
          caption="Distribution of ‘Anomaly’ root-labels in the training data.")

  st.markdown("""
  The countplot below shows the number of ‘Anomaly’ root labels that narratives are tagged with. We observe that 67% of the narratives are tagged with two or more distinct ‘Anomaly’ root labels, making our classification problem a **multiclass, multilabel** one.
 
 The 2.5% of reports with no ‘Anomaly root-label correspond to ‘Other [...]’ labels referring to very specific types of events, that are rare, if not unique, that we decide to exclude from our study.
  """)

  st.image(get_image(img_name = 'cntplot_nbr_labels.png'), 
          caption='Countplot of the number of ‘Anomaly’ root labels that narratives are tagged with. 67% of the narratives are tagged with two or more distinct ‘Anomaly’ root labels, making our classification problem a multiclass, multilabel one.')