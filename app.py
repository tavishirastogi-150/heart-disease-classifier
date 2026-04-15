import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

st.title("❤️ Heart Disease Prediction")

st.write("Enter patient details:")

features = []
for i in range(13):  # dataset has 13 features
    val = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(val)

if st.button("Predict"):
    data = np.array(features).reshape(1, -1)
    data = scaler.transform(data)
    result = model.predict(data)

    if result[0] == 1:
        st.error("Heart Disease Risk ❤️")
    else:
        st.success("No Heart Disease ✅")