import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Titanic Survival Prediction App 🚢")

Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
Parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare", min_value=0.0, value=32.2)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

Sex = 1 if Sex == 'male' else 0
Embarked_map = {'S': 1, 'C': 0, 'Q': 2}
Embarked = Embarked_map[Embarked]

input_data = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                          columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

scaled_input = scaler.transform(input_data)

prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

if prediction[0] == 1:
    st.success("🎉 The passenger would have survived!")
else:
    st.error("💀 The passenger would not have survived.")

st.write("Survival Probability:", round(prediction_proba[0][1]*100, 2), "%")
