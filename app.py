import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Page Configuration
st.set_page_config(
  page_title="Titanic Survival Prediction",
  page_icon="🚢",
  layout="centered"
)

# Background Image
with open("images/titanic_bg.jpg", "rb") as f:
  encoded_string = base64.b64encode(f.read())
  
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpg;base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .header-box {{
        background-color: rgba(255, 255, 255, 0.92);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 20px;
    }}
    .result-box {{
        background-color: rgba(255,255,255,0.88);
        padding: 15px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load Model and Scaler
with open ("models/logistic_model.pkl", "rb") as f:
  model = pickle.load(f)

with open ("models/scaler.pkl", "rb") as f:
  scaler = pickle.load(f)
  
#App Header
st.markdown(
    """
    <div class="header-box">
        <h1 style="color:#1f77b4; margin-bottom:10px;">🚢 Titanic Survival Prediction</h1>
        <p style="font-size:16px; margin:0;">Predict whether a passenger would have survived the Titanic disaster based on input features.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# User Input
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])

col1, col2 = st.columns(2)

with col1:
    Sex = st.selectbox("Sex", ["male", "female"])
    Siblings = st.number_input("Number of Siblings Aboard", min_value=0, max_value=10, value=0)
    Parents = st.number_input("Number of Parents Aboard", min_value=0, max_value=10, value=0)

with col2:
    Age = st.number_input("Age", min_value=0, max_value=100, value=25)
    Spouses = st.number_input("Number of Spouses Aboard", min_value=0, max_value=5, value=0)
    Children = st.number_input("Number of Children Aboard", min_value=0, max_value=10, value=0)

Fare = st.number_input("Fare", min_value=0.0, value=32.2)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Data Preprocessing
SibSp = Siblings + Spouses
Parch = Parents + Children

sex_encoded = 1 if Sex == 'male' else 0
embarked_map = {"S": 1, "C": 0, "Q": 2}
embarked_encoded = embarked_map[Embarked]

# Create input dataframe
input_data = pd.DataFrame(
  [[Pclass, sex_encoded, Age, SibSp, Parch, Fare, embarked_encoded]],
  columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
)

# Scale input
scaled_input = scaler.transform(input_data)

# Prediction
prediction = model.predict(scaled_input)[0]
prediction_proba = model.predict_proba(scaled_input)[0][1]

# Output
if prediction == 1:
    result_text = "<p style='color:green; font-size:18px; font-weight:bold;'> The passenger survived</p>"
else:
    result_text = "<p style='color:red; font-size:18px; font-weight:bold;'> The passenger did not survive</p>"

st.markdown(
    f"""
    <div class="result-box">
        {result_text}
        <p style="font-size:16px; margin-top:10px;">
            <b>Survival Probability:</b> {round(prediction_proba*100, 2)}%
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
