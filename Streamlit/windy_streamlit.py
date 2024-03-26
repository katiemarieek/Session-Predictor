import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import pickle

st.set_page_config(page_title="Session Predictor",
                   page_icon=":ocean:",
                   layout="centered")

# banner
st.image("https://images.pexels.com/photos/2745761/pexels-photo-2745761.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1", use_column_width=True)

# background colour
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color:#92D9D0;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# title
st.title("Session Predictor - Vassiliki")

st.markdown("See if you'll be bobbing, planing or flying!")

st.markdown("Input results for today's forecast to predict your session. Model trained on 6 years of historical data from Vassiliki Bay, finding the link between forecast predictors and the elusive Eric. For best results, use data from the [Windguru](https://www.windguru.cz/49202) website.")

def windy():
    '''function for producing a prediction on session quality from a given set of data, using logreg.'''
    # Take inputs
    ws_12 = st.number_input("Input Wind Speed (in knots) at 12.00", 0, 100)
    ws_15 = st.number_input("Input Wind Speed (in knots) at 16.00", 0, 100)
    ws_18 = st.number_input("Input Wind Speed (in knots) at 18.00", 0, 100)
    ws_21 = st.number_input("Input Wind Speed (in knots) at 20.00", 0, 100)
    wg_12 = st.number_input("Input Wind Gusts (in knots) at 12.00", 0, 100)
    wg_15 = st.number_input("Input Wind Gusts (in knots) at 16.00", 0, 100)
    t_12 = st.number_input("Input Temperature (in Celcius) at 12.00", 0, 100)
    t_15 = st.number_input("Input Temperature (in Celcius) at 16.00", 0, 100)
    t_18 = st.number_input("Input Temperature (in Celcius) at 18.00", 0, 100)
    cc_09 = st.number_input("Input Cloud Cover (High) at 10.00", 0, 100)
    cc_12 = st.number_input("Input Cloud Cover (High) at 12.00", 0, 100)
    cc_15 = st.number_input("Input Cloud Cover (High) at 16.00", 0, 100)
    
    allowed_directions = ['N', 'NNE', 'NE', 'NEE', 'E', 'SEE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'SWW', 
                         'W', 'NWW', 'NW', 'NNW']
    wd_12 = st.selectbox("Please select the Wind Direction at 12.00", allowed_directions)
    wd_15 = st.selectbox("Please select the Wind Direction at 16.00", allowed_directions)
    wd_18 = st.selectbox("Please select the Wind Direction at 18.00", allowed_directions)
    
    # Map directions
    mapping = {'N': 180, 'NNE': 203, 'NE': 225, 'NEE': 248, 'E': 270, 'SEE': 293, 'SE': 315, 'SSE': 338, 'S':       360, 'SSW': 383, 'SW':405, 'SWW': 428, 'W': 450, 'NWW': 472, 'NW': 495, 'NNW': 518}
    wd_12 = mapping[wd_12]
    wd_15 = mapping[wd_15]
    wd_18 = mapping[wd_18]
    
    # Create dataframe
    features = ['ws_12', 'ws_15', 'ws_18', 'ws_21', 'wg_12', 'wg_15', 't_12', 't_15', 't_18', 'cc_09', 'cc_12',     'cc_15', 'wd_12', 'wd_15', 'wd_18']
    values = [ws_12, ws_15, ws_18, ws_21, wg_12, wg_15, t_12, t_15, t_18, cc_09, cc_12, cc_15, wd_12, wd_15,         wd_18]

    data = {feature: value for feature, value in zip(features, values)}
    df = pd.DataFrame(data, index=[0])  
    
    # Load LogReg
    with open('Streamlit/logreg.sav', 'rb') as file: 
        logreg = pickle.load(file) 
    file.close() 
    
    if st.button("Predict my session"):
    # Make prediction
        df[['prob_0', 'prob_1', 'prob_2']] = logreg.predict_proba(df)
        df['y_pred'] = np.where(df['prob_0']>0.47, 0, np.where(df['prob_1']>0.47, 1, 2))

        if df['y_pred'][0] == 0: 
            y_pred_map = 'bobbing'
        elif df['y_pred'][0] == 1: 
            y_pred_map = 'planing'
        elif df['y_pred'][0] == 2: 
            y_pred_map = 'flying'
        st.write(f"Session Prediction : There is a 70% chance you'll be {y_pred_map} today.")
        if df['y_pred'][0] == 0:
            st.write(f"It's not looking like Eric will show up today, so why not spend the afternoon working on your light wind freestyle? \nIf you fancy a challenge, have a look at [this youtube tutorial](https://www.youtube.com/watch?v=TmjiKD8AfDk) to try your first clew-first helitack, clew-first upwind 360, improved sail stall, upwind 360 diablo or duck tack!")
        elif df['y_pred'][1] == 0:
            st.write("Eric is coming! \nWhether it's a big kit blast or fast and furious you're due some windsurfing action today, so keep an eye on those flags.")
        elif df['y_pred'][2] == 0:
            st.write("Batten down the hatches, it's going to be a big one! If you've been waiting for a chance to try some high wind freestyle, today may be your day. Remember, if the weather looks changeable, keep your kit comfortable and your runs short. Good luck!")
windy()  

st.markdown("[Like this app? Donate via Ko-fi to fuel more caffeine driven data projects](https://ko-fi.com/katie42)")
