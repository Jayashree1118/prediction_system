#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 18:12:55 2024

@author: murali
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib

# Load scaler and models
scaler = joblib.load('/Users/murali/Desktop/multiple disease prediction system/scaler.pkl')
diabetes_model = pickle.load(open('/Users/murali/Desktop/multiple disease prediction system/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('/Users/murali/Desktop/multiple disease prediction system/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('/Users/murali/Desktop/multiple disease prediction system/saved_models/parkinsons_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System', 
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           icons=['activity', 'heart-pulse-fill', 'person-arms-up'],
                           default_index=0)

# Add Bootstrap Icons CSS
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.8.1/font/bootstrap-icons.min.css">
    <style>
    .tooltip {
      position: relative;
      display: inline-block;
      cursor: pointer;
    }
    .tooltip .tooltiptext {
      visibility: hidden;
      width: 200px;
      background-color: black;
      color: #fff;
      text-align: center;
      border-radius: 5px;
      padding: 5px 0;
      position: absolute;
      z-index: 1;
      bottom: 100%;
      left: 50%;
      margin-left: -100px;
      opacity: 0;
      transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to add eye icon with tooltip
def add_eye_icon(field_name, info, key):
    st.markdown(f"""
    <div class="tooltip">
      <i class="bi bi-eye-fill"></i>
      <span class="tooltiptext">{info}</span>
    </div>
    """, unsafe_allow_html=True)
    return st.text_input(field_name, key=key)

# Diabetes prediction page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = add_eye_icon('Number of Pregnancies', 'Number of times pregnant(0 if male)', key='Pregnancies')
    with col2:
        Glucose = add_eye_icon('Glucose Level', 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test(mg/dL)', key='Glucose')
    with col3:
        BloodPressure = add_eye_icon('Blood Pressure Value', 'Diastolic blood pressure (mm Hg)', key='BloodPressure')
    with col1:
        SkinThickness = add_eye_icon('Skin Thickness Value', 'Triceps skin fold thickness (mm)', key='SkinThickness')
    with col2:
        Insulin = add_eye_icon('Insulin Level', '2-Hour serum insulin (IU/ml)', key='Insulin')
    with col3:
        BMI = add_eye_icon('BMI Value', 'Body mass index (weight in kg/(height in m)^2)', key='BMI')
    with col1:
        DiabetesPedigreeFunction = add_eye_icon('Diabetes Pedigree Function Value', 'normal range- 0.08 to 2.42', key='DiabetesPedigreeFunction')
    with col2:
        Age = add_eye_icon('Age of the Person', 'Age (years)', key='Age')
    st.write("If you don't understand what type of input has to be given, please click on the link below to access the user guide.")
    st.markdown("[User Guide](https://drive.google.com/file/d/19RxIG4bSyW6PNgHGx6h9-OBf2yjcTU5c/view?usp=sharing)")

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to the appropriate data type
            Pregnancies = float(Pregnancies)
            Glucose = float(Glucose)
            BloodPressure = float(BloodPressure)
            SkinThickness = float(SkinThickness)
            Insulin = float(Insulin)
            BMI = float(BMI)
            DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
            Age = float(Age)

            # Prepare the input data
            input_data = np.asarray([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            input_data_reshaped = input_data.reshape(1, -1)

            # Scale the input data using the loaded scaler
            input_data_scaled = scaler.transform(input_data_reshaped)

            # Make prediction
            diab_prediction = diabetes_model.predict(input_data_scaled)

            if diab_prediction[0] >= 0.5:
                diab_diagnosis = 'The Person is Diabetic'
            else:
                diab_diagnosis = 'The Person is not Diabetic'
        except ValueError:
            diab_diagnosis = 'Please enter valid numeric values for all fields.'


    st.success(diab_diagnosis)

# Heart disease prediction page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using CNN')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = add_eye_icon('Age', 'Age(in years)', key='age')
    with col2:
        sex = add_eye_icon('Sex', 'Gender of the person (1 = male; 0 = female)', key='sex')
    with col3:
        cp = add_eye_icon('Chest Pain types', 'Chest pain type (0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic)', key='cp')
    with col1:
        trestbps = add_eye_icon('Resting Blood Pressure', 'Resting blood pressure (mm Hg)', key='trestbps')
    with col2:
        chol = add_eye_icon('Serum Cholesterol in mg/dl', 'Serum cholesterol(mg/dl)', key='chol')
    with col3:
        fbs = add_eye_icon('Fasting Blood Sugar > 120 mg/dl', 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)', key='fbs')
    with col1:
        restecg = add_eye_icon('Resting Electrocardiographic results', 'Resting electrocardiographic results (0 = normal; 1 = having ST-T wave abnormality; 2 = showing probable or definite left ventricular hypertrophy by Estes\' criteria)', key='restecg')
    with col2:
        thalach = add_eye_icon('Maximum Heart Rate achieved', 'Maximum heart rate achieved(bpm)', key='thalach')
    with col3:
        exang = add_eye_icon('Exercise Induced Angina', 'Exercise induced angina (1 = yes; 0 = no)', key='exang')
    with col1:
        oldpeak = add_eye_icon('ST depression induced by exercise', 'ST depression(risk if greater than 6.2)', key='oldpeak')
    with col2:
        slope = add_eye_icon('Slope of the peak exercise ST segment', 'The slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping)', key='slope')
    with col3:
        ca = add_eye_icon('Major vessels colored by fluoroscopy', 'Number of major vessels (0-3) colored by fluoroscopy', key='ca')
    with col1:
        thal = add_eye_icon('Thalassemia' , 'Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)', key='thal')

    st.write("If you don't understand what type of input has to be given, please click on the link below to access the user guide.")
    st.markdown("[User Guide](https://drive.google.com/file/d/19RxIG4bSyW6PNgHGx6h9-OBf2yjcTU5c/view?usp=sharing)")

    # Code for Prediction
    heart_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] != 1:
            heart_diagnosis = 'The person does not have any heart disease'
        else:
            heart_diagnosis = 'The person is having heart disease'

    st.success(heart_diagnosis)

# Parkinson's prediction page
if selected == 'Parkinsons Prediction':
    st.title('Parkinsons Prediction using ML')

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = add_eye_icon("MDVP-Fo", "Average vocal fundamental frequency(Hz)", key='fo')
    with col2:
        fhi = add_eye_icon('MDVP-Fhi', 'Maximum vocal fundamental frequency(Hz)', key='fhi')
    with col3:
        flo = add_eye_icon('MDVP-Flo', 'Minimum vocal fundamental frequency(Hz)', key='flo')
    with col4:
        Jitter_percent = add_eye_icon('MDVP-Jitter(%)', 'Short-term perturbation in frequency (percentage)', key='Jitter_percent')
    with col5:
        Jitter_Abs = add_eye_icon('MDVP-Jitter(Abs)', 'Short-term perturbation in frequency (Abs)', key='Jitter_Abs')
    with col1:
        RAP = add_eye_icon('MDVP-RAP', 'Relative amplitude perturbation(>0.02-high risk)', key='RAP')
    with col2:
        PPQ = add_eye_icon('MDVP-PPQ', 'Five-point period perturbation quotient(>0.02-high risk)', key='PPQ')
    with col3:
        DDP = add_eye_icon('Jitter-DDP', 'Average absolute difference of differences between cycles(>0.06 -high risk)', key='DDP')
    with col4:
        Shimmer = add_eye_icon('MDVP-Shimmer', 'Shimmer(>0.15 -high risk)', key='Shimmer')
    with col5:
        Shimmer_dB = add_eye_icon('MDVP-Shimmer(dB)', 'Shimmer in dB(>2 -high risk)', key='Shimmer_dB')
    with col1:
        APQ3 = add_eye_icon('Shimmer-APQ3', 'Three-point amplitude perturbation quotient(>0.1 -high risk)', key='APQ3')
    with col2:
        APQ5 = add_eye_icon('Shimmer-APQ5', 'Five-point amplitude perturbation quotient(>0.1 -high risk)', key='APQ5')
    with col3:
        APQ = add_eye_icon('MDVP-APQ', 'Amplitude perturbation quotient(>0.15 -high risk)', key='APQ')
    with col4:
        DDA = add_eye_icon('Shimmer-DDA', 'Average absolute difference of differences between cycles(>0.2 -high risk)', key='DDA')
    with col5:
        NHR = add_eye_icon('NHR', 'Noise-to-harmonics ratio(>0.5 -high risk)', key='NHR')
    with col1:
        HNR = add_eye_icon('HNR', 'Harmonics-to-noise ratio(normal range- 10 to 30)', key='HNR')
    with col2:
        RPDE = add_eye_icon('RPDE', 'Recurrence period density entropy(>1 -high risk)', key='RPDE')
    with col3:
        DFA = add_eye_icon('DFA', 'Detrended fluctuation analysis(normal range- 0.5 to 1)', key='DFA')
    with col4:
        spread1 = add_eye_icon('spread1', 'Nonlinear measure of fundamental frequency variation(>2 -high risk)', key='spread1')
    with col5:
        spread2 = add_eye_icon('spread2', 'Nonlinear measure of fundamental frequency variation(>2 -high risk)', key='spread2')
    with col1:
        D2 = add_eye_icon('D2', 'Correlation dimension(normal range: -10 to -1)', key='D2')
    with col2:
        PPE = add_eye_icon('PPE', 'Dynamical complexity measure(>1 -high risk)', key='PPE')

    st.write("If you don't understand what type of input has to be given, please click on the link below to access the user guide.")
    st.markdown("[User Guide](https://drive.google.com/file/d/19RxIG4bSyW6PNgHGx6h9-OBf2yjcTU5c/view?usp=sharing)")

    parkinsons_diagnosis = ''

    if st.button('Parkinsons Test Result'):
        try:
            # Convert inputs to the appropriate data type
            fo = float(fo)
            fhi = float(fhi)
            flo = float(flo)
            Jitter_percent = float(Jitter_percent)
            Jitter_Abs = float(Jitter_Abs)
            RAP = float(RAP)
            PPQ = float(PPQ)
            DDP = float(DDP)
            Shimmer = float(Shimmer)
            Shimmer_dB = float(Shimmer_dB)
            APQ3 = float(APQ3)
            APQ5 = float(APQ5)
            APQ = float(APQ)
            DDA = float(DDA)
            NHR = float(NHR)
            HNR = float(HNR)
            RPDE = float(RPDE)
            DFA = float(DFA)
            spread1 = float(spread1)
            spread2 = float(spread2)
            D2 = float(D2)
            PPE = float(PPE)

            # Prepare the input data
            input_data = np.asarray([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
            input_data_reshaped = input_data.reshape(1, -1)

            # Make prediction
            parkinsons_prediction = parkinsons_model.predict(input_data_reshaped)

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'The person has Parkinsons disease'
            else:
                parkinsons_diagnosis = 'The person does not have Parkinsons disease'
        except ValueError:
            parkinsons_diagnosis = 'Please enter valid numeric values for all fields.'

    st.success(parkinsons_diagnosis)
