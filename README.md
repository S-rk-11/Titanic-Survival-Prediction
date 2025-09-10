# Titanic Survival Prediction

## Project Overview
This project predicts whether a passenger would have survived the Titanic disaster using a Logistic Regression model.  
It demonstrates the machine learning workflow: preprocessing, model training, evaluation, and deployment with Streamlit.  

## Project Structure
titanic-survival-prediction/  
├─ models/ 
│  └─ logistic_model.pkl 
│  └─ scaler.pkl 
├─ data/ 
│  └─ Titanic.csv 
├─ app.py 
├─ requirements.txt 
├─ README.md 

## Workflow
- **Data Preprocessing**: handling missing values, encoding categorical variables, scaling  
- **Modeling**: Logistic Regression  
- **Evaluation**: Accuracy, confusion matrix, classification report  
- **Deployment**: Streamlit app for interactive survival prediction

## Results
- Model Used: **Logistic Regression**  
- Accuracy: ~ **80%**  

## Next Steps
- Improve model with advanced algorithms (Random Forest, XGBoost)  
- Hyperparameter tuning for better accuracy  
- Deploy app as a web app on **Streamlit Cloud** 

## Installation
Clone this repository and install dependencies:
(```bash)

git clone https://github.com/S-rk-11/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
streamlit run app.py

Author
Shivani Kalghatgi

LinkedIn:
Kaggle:
Email: shivanikalghatgi@gmail.com
