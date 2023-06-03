import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder

st.subheader("Application qui prédit l'accord du crédit")

# Collecter le profil d'entrée
st.sidebar.header("Les caractéristiques du client")

def client_car():

    Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    Married = st.sidebar.selectbox("Married", ("Yes", "No"))
    Dependents = st.sidebar.selectbox("Dependents", ("0", "1", "2", "3+"))
    Education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
    Self_Employed = st.sidebar.selectbox("Self_Employed", ("Yes", "No"))
    ApplicantIncome = st.sidebar.slider("Income_client", 150, 50000, 5000)
    CoapplicantIncome = st.sidebar.slider("Income_conjoint", 0, 50000, 5000)
    LoanAmount = st.sidebar.slider("Montant du crédit", 9.0, 700.0, 200.0)
    Loan_Amount_Term = st.sidebar.selectbox("Durée", (360.0, 120.0, 240.0, 180.0, 60.0, 300.0, 36.0, 84.0, 12.0))
    Credit_History = st.sidebar.selectbox("Credit_History", (1.0, 0.0))
    Property_Area = st.sidebar.selectbox("Property_Area", ("Urban", "Rural", "Semiurban"))

    data = {
    'Gender' : Gender,
    'Married': Married,
    'Dependents': Dependents, 
    'Education' : Education,
    'Self_Employed' : Self_Employed,
    'ApplicantIncome' : ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome, 
    'LoanAmount' : LoanAmount,
    'Loan_Amount_Term' : Loan_Amount_Term,
    'Credit_History' : Credit_History,
    'Property_Area':Property_Area}
    
    profil = pd.DataFrame(data, index=[0])
    return profil

df = client_car()
st.write(df)       

# Transformer les données pour le modèle
train = pd.read_csv("train.csv")
train.drop(["Loan_Status","Loan_ID"], axis=1, inplace=True)

df = pd.concat([df,train], axis=0)


df = pd.get_dummies(df, drop_first=True)

df = df[:1]  

# Importer le modèle
model_loaded = joblib.load(filename="loan_model.pkl")

# Appliquer le modèle

pred = model_loaded.predict(df)

if pred == "Y":
    st.write("Le crédit est accordé")
else:
    st.write("Le crédit n'est pas accordé")