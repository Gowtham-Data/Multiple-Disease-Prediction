import streamlit as st
import pandas as pd
import joblib

# ----------------- Load Models -----------------
kidney_model = joblib.load('kidney_model.pkl')
parkinsons_model = joblib.load('parkinsons_model.pkl')
liver_model = joblib.load('liver_model.pkl')

st.title("🧬 Multiple Disease Prediction App")

# Select Disease
disease = st.selectbox("Select Disease:", 
                       ["Kidney Disease", "Parkinson's Disease", "Liver Disease"])


# ✅ KIDNEY DISEASE (5 Features - Matches Training)
if disease == "Kidney Disease":
    st.header("Kidney Disease Prediction")

    bp = st.number_input("Blood Pressure (BP)", 50, 200, 80)
    sg = st.number_input("Specific Gravity (SG)", 1.00, 1.030, 1.01, step=0.001)
    al = st.number_input("Albumin (AL)", 0, 5, 0)
    su = st.number_input("Sugar (SU)", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random (BGR)", 50, 500, 120)

    input_df = pd.DataFrame([[bp, sg, al, su, bgr]],
                            columns=['bp', 'sg', 'al', 'su', 'bgr'])

    if st.button("Predict"):
        pred = kidney_model.predict(input_df)[0]
        prob = kidney_model.predict_proba(input_df)[0][1]
        if pred == 1:
            st.error(f"⚠️ Kidney Disease Detected! (Prob: {prob:.2f})")
        else:
            st.success(f"✅ No Kidney Disease (Prob: {prob:.2f})")


# ✅ PARKINSON’S DISEASE (22 Features - Matches Training)
elif disease == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")

    parkinsons_features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
        'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
        'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    input_values = []
    for feature in parkinsons_features:
        input_values.append(st.number_input(feature, value=0.0))

    input_df = pd.DataFrame([input_values], columns=parkinsons_features)

    if st.button("Predict"):
        pred = parkinsons_model.predict(input_df)[0]
        prob = parkinsons_model.predict_proba(input_df)[0][1]
        if pred == 1:
            st.error(f"⚠️ Parkinson's Disease Detected! (Prob: {prob:.2f})")
        else:
            st.success(f"✅ No Parkinson's Disease (Prob: {prob:.2f})")


# ✅ LIVER DISEASE (10 Features - Matches Training)
elif disease == "Liver Disease":
    st.header("Liver Disease Prediction")

    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", 0.0, 50.0, 1.0)
    direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 20.0, 0.3)
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase", 50, 5000, 200)
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase", 0, 2000, 40)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", 0, 2000, 35)
    total_protiens = st.number_input("Total Proteins", 0.0, 20.0, 6.8)
    albumin = st.number_input("Albumin", 1.0, 10.0, 3.4)
    albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio", 0.0, 3.0, 1.0)

    gender_val = 1 if gender == "Female" else 0

    input_df = pd.DataFrame([[age, gender_val, total_bilirubin, direct_bilirubin,
                              alkaline_phosphotase, alamine_aminotransferase,
                              aspartate_aminotransferase, total_protiens,
                              albumin, albumin_globulin_ratio]],
                            columns=[
                                'Age','Gender','Total_Bilirubin','Direct_Bilirubin',
                                'Alkaline_Phosphotase','Alamine_Aminotransferase',
                                'Aspartate_Aminotransferase','Total_Protiens',
                                'Albumin','Albumin_and_Globulin_Ratio'
                            ])

    if st.button("Predict"):
        pred = liver_model.predict(input_df)[0]
        prob = liver_model.predict_proba(input_df)[0][1]
        if pred == 1:
            st.error(f"⚠️ Liver Disease Detected! (Prob: {prob:.2f})")
        else:
            st.success(f"✅ No Liver Disease (Prob: {prob:.2f})")
