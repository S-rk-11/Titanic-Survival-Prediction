import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model and Scaler
with open ("models/logistic_model.pkl", "rb") as f:
  model = pickle.load(f)

with open ("models/scaler.pkl", "rb") as f:
  scaler = pickle.load(f)
  
#App TItle
st.title("Titanic Survival Prediction")
st.markdown(
  "This app predicts whether a passenger would have survived the Titanic disaster."
  "based on input features."
)

# User Input
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
Parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare", min_value=0.0, value=32.2)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Data Preprocessing
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
    st.success("The passenger **survived**")
else:
    st.error("The passenger did **not survived**.")

st.write(f"**Survival Probability:** {round(prediction_proba *100, 2)}%")
