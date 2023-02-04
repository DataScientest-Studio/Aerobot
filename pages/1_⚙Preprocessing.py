import streamlit as st
import os
from pathlib import Path
import inspect
import numpy as np
from annotated_text import annotated_text
from streamlitpackages import get_img_with_href, get_image

# Configure sidebar
streamlit_home_dir = str(Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))).parents[0])
with st.sidebar:
  st.header("Contact")
  logo_linkedin = get_img_with_href(os.path.join(streamlit_home_dir, 'ressources/linkedin.png'), 'https://www.linkedin.com/in/ioannis-stasinopoulos/', 20)
  st.write(f'''<a href="https://www.linkedin.com/in/ioannis-stasinopoulos/" style="text-decoration: none;color:black">Ioannis STASINOPOULOS</a> {logo_linkedin}''', unsafe_allow_html=True) 
  st.write(f'''<a href="https://www.linkedin.com/in/heleneassir/" style="text-decoration: none;color:black">Hélène ASSIR</a> {logo_linkedin}''', unsafe_allow_html=True) 

st.markdown("""
          # ⚙ Preprocessing
          ## Preparing narratives for Bag-of-words models
          """)

st.markdown("### 🏝 A weekend trip to the Bahamas") # 😎

# Define colors and labels for expressions, contractions
expr = {"color": "#26DBE0", # DST cyan
        "font_color":  "#000000"} 

contr = {"color": "#4529DE",# DST blue
          "font_color": "#FFFFFF"} 

num = {"color": "#A329DE",# DST violet
          "font_color": "#FFFFFF"} 
    
#000000 black
#26DBE0 # DST cyan
#F9CB5E # DST orange
#A329DE # DST violet
#4529DE # DST blue
#E8E3D4 # DST gray (modified)


st.write("We show an example narrative, where **a pilot describes how he landed on \
            a taxiway\* instead of the runway (cf. highlighted text).**")
st.markdown("""
<style>
.small-font {font-size:12px ;}
</style>
<em class="small-font">*taxiways are the 'roads' of an airport that airplanes use on the ground </em>
""", unsafe_allow_html=True)

st.write("You can visualize the narrative either:")
annotated_text("- in its original form")
annotated_text("or")
annotated_text("- with highlighted and translated ", ("CONTRACTIONS", "", contr['color'], contr["font_color"]),
  " , ", ("EXPRESSIONS", "", expr['color'], expr["font_color"])) 
annotated_text(" and ", ("COMPLEX NUMERIC", "", num['color'], num["font_color"]), "information.") 

text_display_form = st.radio(
      '',
      ('Original', 'With annotations'))
if text_display_form == 'Original':
    annotated_text(
    "I WAS RETURNING FROM A WEEKEND TRIP TO THE BAHAMAS.") 
    annotated_text(
      "I WAS NEAR THE COMPLETION OF MY IFR FLT PLAN") 
    annotated_text(
      "WHEN PALM BEACH APCH INITIALLY CLRED ME FOR THE VISUAL RWY13.")
    annotated_text(
      "UPON TURNING BASE AND HANDED OFF TO PALM BEACH TOWER;")
    annotated_text(
      "THE RWY WAS CHANGED AND I WAS CLRED FOR THE VISUAL RWY 9L.")
    annotated_text(
      "AT APPROX ONE (1) MILE; ON FINAL; FROM THE NUMBERS THE TOWER")
    annotated_text(
      "ASKED IF I WOULD MIND SWITCHING TO RWY 9R. I WAS HEARING")
    annotated_text(
    "THE 'CHATTER' FROM THE COMMERCIAL ACFT WAITING AND REQUESTING")
    annotated_text(
    "TO DEPART ON RWY 9L. TRYING TO BE ACCOMMODATING BECAUSE")
    annotated_text(
      "MY ACFT WAS MUCH SLOWER; I AGREED TO SWITCH RWYS.")
    annotated_text(
      "THE TOWER CLRED ME FOR RWY 9R AND TO SIDESLIP FOR MY APCH.")
    annotated_text(
      "AFTER I LANDED I WAS INFORMED THAT")
    annotated_text(
      ("I HAD LANDED ON THE TXWY RATHER THAN THE RWY.", "", "#ebe12d", "#000000")) # black fonts on yellow bgr
    annotated_text(
      "THERE WERE NO AIRPLANES ON EITHER THE TXWY OR RWY 9R.")
    annotated_text(
      "MY CO-PILOT (ALSO A LICENSED IFR PLT) AND I WERE")
    annotated_text(
      "TOTALLY SHOCKED TO LEARN THAT I HAD DONE THIS. NEITHER OF US")
    annotated_text(
      "HAD ANY AWARENESS THAT THERE WAS A PROBLEM UNTIL INFORMED OF SAME.")
    annotated_text(
      "THE ONLY OTHER OBSERVATIONS THAT MIGHT BE RELEVANT WAS THAT")
    annotated_text(
      "RWY 9L IS SUBSTANTIALLY LARGER AND MORE PROMINENT")
    annotated_text(
      "ON FINAL APCH; THE TXWY NEXT TO RWY 9L IS LONGER AND WIDER")
    annotated_text("THAN RWY 9R CAUSING SOME VISUAL CONFUSION;")
    annotated_text(
      "AND THE ATIS WHEN I LANDED WAS 070@15G24KTS")
    annotated_text(
      "CAUSING TURBULENCE ON APCH AND LNDG.")
    annotated_text(
      "MAKING NO EXCUSES FOR THIS INCIDENT; I WAS INFORMED")
    annotated_text(
      "BY TOWER PERSONNEL THAT THIS PARTICULAR INCIDENT")
    annotated_text(
      "DOES OCCUR FREQUENTLY.")

else:
  annotated_text(
    "I WAS RETURNING FROM A WEEKEND TRIP TO THE BAHAMAS.") 
  annotated_text(
    "I WAS NEAR THE COMPLETION OF MY ", ("IFR", "INSTRUMENT FLIGHT RULES", expr['color'], expr['font_color']), ("FLT", "FLIGHT", contr['color'], contr['font_color']), " PLAN") 
  annotated_text(
    "WHEN PALM BEACH ", ("APCH", "APPROACH", contr['color'], contr['font_color']), " INITIALLY ", ("CLRED", "CLEARED", contr['color'], contr["font_color"]), " ME FOR THE VISUAL ", ("RWY13", "text+numeric", num['color'], num['font_color']), ".")
  annotated_text(
    "UPON TURNING BASE AND HANDED OFF TO PALM BEACH TOWER;")
  annotated_text(
    "THE ", ("RWY", "RUNWAY", contr['color'], contr['font_color']), " WAS CHANGED AND I WAS ", ("CLRED", "CLEARED", contr['color'], contr["font_color"]), " FOR THE VISUAL ", ("RWY", "RUNWAY", contr['color'], contr['font_color']), " 9L.")
  annotated_text(
    "AT ", ("APPROX", "APPROXIMATELY", contr['color'], contr['font_color']), " ONE (1) MILE; ON FINAL; FROM THE NUMBERS THE TOWER")
  annotated_text(
    "ASKED IF I WOULD MIND SWITCHING TO ", ("RWY", "RUNWAY", contr['color'], contr['font_color']), " 9R. I WAS HEARING")
  annotated_text(
  "THE 'CHATTER' FROM THE COMMERCIAL ", ("ACFT", "AIRCRAFT", contr['color'], contr['font_color']), " WAITING AND REQUESTING")
  annotated_text(
  "TO DEPART ON ", ("RWY", "RUNWAY", contr['color'], contr['font_color']), " 9L. TRYING TO BE ACCOMMODATING BECAUSE")
  annotated_text(
    "MY ", ("ACFT", "AIRCRAFT", contr['color'], contr['font_color']), " WAS MUCH SLOWER; I AGREED TO SWITCH ", ("RWYS", "RUNWAYS", contr['color'], contr['font_color']), ".")
  annotated_text(
    "THE TOWER ", ("CLRED", "CLEARED", contr['color'], contr["font_color"]), " ME FOR ", ("RWY", "RUNWAY", contr['color'], contr['font_color']), " 9R AND TO SIDESLIP FOR MY ", ("APCH", "APPROACH", contr['color'], contr['font_color']), ".")
  annotated_text(
    "AFTER I LANDED I WAS INFORMED THAT")
  annotated_text(
    "I HAD LANDED ON THE ", ("TXWY", "TAXIWAY", contr['color'], contr['font_color']), " RATHER THAN THE ", ("RWY", "RUNWAY", contr['color'], contr['font_color']), ".")
  annotated_text(
    "THERE WERE NO AIRPLANES ON EITHER THE ", ("TXWY", "TAXIWAY", contr['color'], contr['font_color']), " OR ", ("RWY", "RUNWAY", contr['color'], contr['font_color']), " 9R.")
  annotated_text(
    "MY CO-PILOT (ALSO A LICENSED ", ("IFR", "INSTRUMENT FLIGHT RULES", expr['color'], expr['font_color']), " ", ("PLT", "PILOT", contr['color'], contr['font_color']), ") AND I WERE")
  annotated_text(
    "TOTALLY SHOCKED TO LEARN THAT I HAD DONE THIS. NEITHER OF US")
  annotated_text(
    "HAD ANY AWARENESS THAT THERE WAS A PROBLEM UNTIL INFORMED OF SAME.")
  annotated_text(
    "THE ONLY OTHER OBSERVATIONS THAT MIGHT BE RELEVANT WAS THAT")
  annotated_text(
    "", ("RWY", "RUNWAY", contr['color'], contr['font_color']), " 9L IS SUBSTANTIALLY LARGER AND MORE PROMINENT")
  annotated_text(
    "ON FINAL ", ("APCH", "APPROACH", contr['color'], contr['font_color']), "; THE ", ("TXWY", "TAXIWAY", contr['color'], contr['font_color']), " NEXT TO ", ("RWY", "RUNWAY", contr['color'], contr['font_color']), " 9L IS LONGER AND WIDER")
  annotated_text("THAN ", ("RWY", "RUNWAY", contr['color'], contr['font_color']), " 9R CAUSING SOME VISUAL CONFUSION;")
  annotated_text(
    "AND THE ", ("ATIS", "Automatic Terminal Information Service", expr['color'], expr['font_color'])) 
  annotated_text(
    " WHEN I LANDED WAS ", ("070@15G24KTS", "wind indication", num['color'], num['font_color']))
  annotated_text(
    "CAUSING TURBULENCE ON ", ("APCH", "APPROACH", contr['color'], contr['font_color']), " AND ", ("LNDG", "LANDING", contr['color'], contr['font_color']), ".")
  annotated_text(
    "MAKING NO EXCUSES FOR THIS INCIDENT; I WAS INFORMED")
  annotated_text(
    "BY TOWER PERSONNEL THAT THIS PARTICULAR INCIDENT")
  annotated_text(
    "DOES OCCUR FREQUENTLY.")
st.write(" ") # placeholder


with st.expander("Click to play a game..."):
  st.markdown("""##### Can you find the label(s) assigned to this narrative?
              """)
  st.write("Hint: 'ATC' means 'Air Traffic Control'")

  anomaly_tuple = (
      '01 Deviation / Discrepancy - Procedural',
      '02 Aircraft Equipment',
      '03 Conflict',
      '04 Inflight Event / Encounter',
      '05 ATC Issue',
      '06 Deviation - Altitude',
      '07 Deviation - Track / Heading',
      '08 Ground Event / Encounter',
      '09 Flight Deck / Cabin / Aircraft Event',
      '10 Ground Incursion',
      '11 Airspace Violation',
      '12 Deviation - Speed',
      '13 Ground Excursion',
      '14 No Specific Anomaly Occurred'
      )

  wrong_icon_list = ["🙃", "🙈", "👻", "😝", "😔", "🤕", "😕", "😞", "💩"]
  chkboxlist1 = []
  chkboxlist2 = []
  chkbox01 = st.checkbox(anomaly_tuple[0])
  if chkbox01:
        st.success("Correct! Landing on the taxiway is an obvious deviation from the normal procedure, isn't it? ", icon="✅")
  for i, label in enumerate(anomaly_tuple[1:9]):
    chkboxlist1.append(st.checkbox(label)) 
    if chkboxlist1[i]:
        st.error('This is wrong.', icon=wrong_icon_list[np.random.randint(0, len(wrong_icon_list))])
  chkbox10 = st.checkbox(anomaly_tuple[9])
  if chkbox10:
        st.success('Correct! A ground *incursion* occurs \
          when an aircraft inadvertently touches the ground.', icon="✅")
  for i, label in enumerate(anomaly_tuple[10:12]):
    chkboxlist2.append(st.checkbox(label)) 
    if chkboxlist2[i]:
        st.error('This is wrong.', icon=wrong_icon_list[np.random.randint(0, len(wrong_icon_list))])
  chkbox13 = st.checkbox(anomaly_tuple[12])
  if chkbox13:
    st.error('This is wrong: A runway *excursion* \
          is an incident, in which an aircraft leaves the runway \
            surface / makes an inappropriate exit from the runway.', icon=wrong_icon_list[np.random.randint(0, len(wrong_icon_list))])
  chkbox14 = st.checkbox(anomaly_tuple[-1])
  if chkbox14:
      st.error('This is wrong.', icon=wrong_icon_list[np.random.randint(0, len(wrong_icon_list))])

  if chkbox01 + chkbox10 == 1:
    st.warning("Well done; but that's not all...", icon="👍")
  elif chkbox01 + chkbox10 == 2:
    st.success("Bravo! You have found the 2 labels assigned to this narrative!", icon="🤩")
    st.balloons()
  elif chkbox01 + chkbox10 == 0 and (sum(chkboxlist1) + sum(chkboxlist2) + chkbox13 + chkbox14) != 0:
    st.warning("Keep searching...", icon = "🧐")

  if (sum(chkboxlist1) + sum(chkboxlist2) + chkbox01 + chkbox10 + chkbox13 + chkbox14) == 0:
    st.warning("C'mon, play the game! Just click on the boxes...", icon = "🥺")

st.markdown("#### Narrative preprocessing into 3 forms ")
st.write("We pre-process the narratives to keep the following 3 \
  forms and illustrate this on an excerpt from the narrative above. We have expanded\
    the expression 'Instrument Flight Rules' in order to illustrate what happens during preprocessing, when it\
      is present in its explicit form:")

st.markdown("###### 1. Original ")
st.markdown("""<ins>*Example:*</ins>""", unsafe_allow_html=True)
st.markdown("""
        <body>
        <em>
        I WAS NEAR THE COMPLETION OF MY <span style="background-color:#F9CB5E";>INSTRUMENT FLIGHT RULES </span> 
        &nbsp
        <span style="background-color:#4529DE;color:#FFFFFF">FLT </span>  
        &nbspPLAN WHEN PALM BEACH 
        <span style="background-color:#4529DE;color:#FFFFFF">APCH </span> 
        &nbspINITIALLY 
        <span style="background-color:#4529DE;color:#FFFFFF">CLRED </span>  
        &nbspME FOR THE VISUAL 
        <span style="background-color:#A329DE;color:#FFFFFF">RWY13 </span>
        </em>
        </body>
        """, unsafe_allow_html=True)
st.write(" ")# placeholder
st.write(" ")# placeholder

st.markdown("###### 2. 'NLP': standard Natural Language Preprocessing")
st.write("We lower-case the text, apply tokenization, stop word filtering and stemming:")
st.markdown("""<ins>*Example:*</ins>""", unsafe_allow_html=True)
st.markdown("""
        <body>
        <em>
        'near', 'complet', 
        &nbsp
        <span style="background-color:#F9CB5E";>'instrument'</span> 
        ,&nbsp
        <span style="background-color:#F9CB5E";>'flight'</span> 
        ,&nbsp
        <span style="background-color:#F9CB5E";>'rule'</span> 
        &nbsp
        <span style="background-color:#4529DE;color:#FFFFFF">'flt', </span>  
        &nbsp'plan', 'palm', 'beach', 
        &nbsp
        <span style="background-color:#4529DE;color:#FFFFFF">'apch', </span> 
        &nbsp
        'initi', 
        &nbsp
        <span style="background-color:#4529DE;color:#FFFFFF">'clred', </span>  
        &nbsp
        'visual', 
        <span style="background-color:#A329DE;color:#FFFFFF">'rwy13'</span>
        </em>
        </body>
        """, unsafe_allow_html=True)
st.write(" ")# placeholder
st.write(" ")# placeholder

st.markdown("###### 3. 'TLP': NLP + specific Technical Language Preprocessing")
st.markdown("""
In *addition* to the NLP preprocessing, we perform the following substitutions:
- Contractions by their full-text form, e.g. 'FLT' ⇒ flight
- Expressions by their acronyms, e.g. 'INSTRUMENT FLIGHT RULES' ⇒ IFR
and use Regular Expressions (RegEx) to separate numbers in the case of quantitative information mixed with text (e.g. 'RWY13'), 
which means ‘runway oriented at 130 degrees relative to the north', i.e. South East.
""")
st.markdown("""<ins>*Example:*</ins>""", unsafe_allow_html=True)
st.markdown("""
        <body>
        <em>
        'near', 'complet', 
        &nbsp
        <span style="background-color:#F9CB5E";>'ifr',</span> 
        &nbsp
        <span style="background-color:#4529DE;color:#FFFFFF">'flight', </span>  
        &nbsp'plan', 'palm', 'beach', 
        &nbsp
        <span style="background-color:#4529DE;color:#FFFFFF">'approach', </span> 
        &nbsp
        'initi', 
        &nbsp
        <span style="background-color:#4529DE;color:#FFFFFF">'clear', </span>  
        &nbsp
        'visual', 
        <span style="background-color:#A329DE;color:#FFFFFF">'runway'</span>
        ,&nbsp  
        <span style="background-color:#A329DE;color:#FFFFFF">'13'</span>
        </em>
        </body>
        """, unsafe_allow_html=True)

st.write(" ")# placeholder

st.markdown("""### Narrative length
The figure below shows the impact of the preprocessing on the narrative length. We see that it is marginal.

We count the number of tokens in each narrative, after tokenization and stemming. 
We superpose the histograms of the stemmed 'Original data' and the 'TLP'-preprocessed, stemmed narratives in blue and orange, respectively. 
Given the slight transparency of the histograms and their almost perfect overlap, the result is gray. 
Non-overlapping regions appear orange (some are visible near the peak) or blue (none is visible).
  """)
st.image(get_image(img_name = 'narrative_length.png'), 
        width = 250,
        caption='Impact of the preprocessing on the narrative length: it is marginal as shown by the overlap of the blue and orange semi-transparent bars, that yields gray.')

st.markdown("""
We deduce that we have performed mostly **substitutions** (e.g. ACFT → aircraft, FLT → flight) that did not affect the narrative length.
Although we reduced terms such as ‘Flight Attendant’ to ‘FA’, this did not significantly impact the narrative length. Note: ‘Air Traffic Control’ was mostly found as ‘ATC’ in the first place.

To support these statements, we now plot the top 15 tokens with most occurrences before and after preprocessing. 
As we are looking at absolute frequency, the frequency of some tokens, e.g. ‘aircraft’ exceeds by far the corpus length.
""")
st.image(get_image(img_name = 'abs_frequ.png'), 
        # width = 250,
        caption='Top-15 tokens with the highest occurrence (absolute frequency, not document frequency) before (top) and after (bottom) preprocessing the narratives. The x-axis is scaled identically, for comparison. Note that the frequency exceeds by far the corpus length. The blue arrows point at the conversion of ‘acft’ in ‘aircraft’, reflected in the drastic increase of the latter’s counts in the bottom plot.')

st.markdown("""
We observe that 
- ‘acft’ has been replaced by ‘aircraft’
- ‘ft’ is replaced by ‘feet’
- ‘rwi’ is the stemmed form of ‘rwy’. 
""")