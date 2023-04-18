import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json


# Load All Files
with open("best_model_dtr.pkl", "rb") as file_1:
    best_model_dtr = pickle.load(file_1)

with open("list_num_cols.txt", "r") as file_2:
    list_num_cols = json.load(file_2)

with open("list_catn_cols.txt", "r") as file_3:
    list_catn_cols = json.load(file_3)


def run():
    with st.form(key="Form_Loan_Status"):
        name = st.text_input("Name", value="")
        Gender = st.selectbox("Gender", ("Male", "Female"))
        Married = st.selectbox("Married", ("Yes", "No"))
        Dependents = st.selectbox("Dependents", ("0", "1", '2', '3+'))
        Self_Employed = st.selectbox("Self_Employed", ("Yes", "No"))
        Credit_History = st.selectbox("Credit_History", (1., 0.))
        Property_Area = st.selectbox("Property_Area", ("Rural", "Urban", 'Semiurban'))
        Education = st.selectbox("Education", ("Graduate", "Not Graduate"))
        st.markdown("---")
        ApplicantIncome = st.number_input("ApplicantIncome", min_value=0, max_value=1000000, value=2500, help="Applicant Income")
        CoapplicantIncome = st.number_input("CoapplicantIncome", min_value=0, max_value=1000000, value=2500, help="Co Applicant Income")
        LoanAmount = st.number_input("LoanAmount", min_value=0, max_value=1000000, value=100, help="Loant Amount")
        Loan_Amount_Term = st.number_input("Loan_Amount_Term", min_value=0, max_value=1000, value=120, help="Loant Amount Term")

        submitted = st.form_submit_button("Predict")

    data_inf = {
        'name': name,
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Self_Employed': Self_Employed,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area,
        'Education': Education,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term
    }
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        num_cols = list_num_cols
        catn_cols = list_catn_cols

        y_pred = best_model_dtr.predict(data_inf) # Jalankan model
        if y_pred == True:
            st.write("# Approved")
        else:
            st.write("# Not Approved")

if __name__ == "__app__":
    run()