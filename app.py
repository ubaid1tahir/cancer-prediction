import numpy as np
import joblib
from joblib import load
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# Load your pre-trained model
model = load(r"C:\Users\mu499\Downloads\Cancer-Prediction-Model (Only Useful Features).jbl")

def main():
    st.title('Breast Cancer Diagnosis Predictor')
    st.write('Enter the values for the following features to predict breast cancer diagnosis:')
    
    # Define input fields and sliders for each feature
    radius_mean = st.slider('Radius Mean', min_value=0.0, max_value=30.0, value=15.0, step=0.1)
    perimeter_mean = st.slider('Perimeter Mean', min_value=0.0, max_value=200.0, value=100.0, step=1.0)
    area_mean = st.slider('Area Mean', min_value=0.0, max_value=2500.0, value=500.0, step=10.0)
    compactness_mean = st.slider('Compactness Mean', min_value=0.0, max_value=0.5, value=0.2, step=0.01)
    concavity_mean = st.slider('Concavity Mean', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    concave_points_mean = st.slider('Concave Points Mean', min_value=0.0, max_value=0.3, value=0.15, step=0.01)
    radius_se = st.slider('Radius SE', min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    perimeter_se = st.slider('Perimeter SE', min_value=0.0, max_value=30.0, value=15.0, step=0.5)
    area_se = st.slider('Area SE', min_value=0.0, max_value=800.0, value=200.0, step=5.0)
    radius_worst = st.slider('Radius Worst', min_value=0.0, max_value=40.0, value=20.0, step=0.5)
    perimeter_worst = st.slider('Perimeter Worst', min_value=0.0, max_value=300.0, value=150.0, step=1.0)
    area_worst = st.slider('Area Worst', min_value=0.0, max_value=4500.0, value=1000.0, step=10.0)
    compactness_worst = st.slider('Compactness Worst', min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    concavity_worst = st.slider('Concavity Worst', min_value=0.0, max_value=1.0, value=0.4, step=0.01)
    concave_points_worst = st.slider('Concave Points Worst', min_value=0.0, max_value=0.5, value=0.2, step=0.01)

    # Option to use space-separated input
    use_input = st.radio("Use Space-Separated Input", ("No", "Yes"))
    
    if use_input == "Yes":
        input_values = st.text_input("Enter values separated by spaces")
        if input_values:
            try:
                values = list(map(float, input_values.split()))
                if len(values) == 15:
                    radius_mean = values[0]
                    perimeter_mean = values[1]
                    area_mean = values[2]
                    compactness_mean = values[3]
                    concavity_mean = values[4]
                    concave_points_mean = values[5]
                    radius_se = values[6]
                    perimeter_se = values[7]
                    area_se = values[8]
                    radius_worst = values[9]
                    perimeter_worst = values[10]
                    area_worst = values[11]
                    compactness_worst = values[12]
                    concavity_worst = values[13]
                    concave_points_worst = values[14]
                else:
                    st.warning("Please enter exactly 15 values separated by spaces.")
            except ValueError:
                st.error("Invalid input. Please enter numeric values separated by spaces.")
    
    # Prediction button
    if st.button('Predict Diagnosis'):
        prediction = predict_diagnosis(radius_mean, perimeter_mean, area_mean, compactness_mean,
                                       concavity_mean, concave_points_mean, radius_se, perimeter_se,
                                       area_se, radius_worst, perimeter_worst, area_worst,
                                       compactness_worst, concavity_worst, concave_points_worst)
        
        st.write(f'Prediction: {prediction}')

def predict_diagnosis(radius_mean, perimeter_mean, area_mean, compactness_mean,
                      concavity_mean, concave_points_mean, radius_se, perimeter_se,
                      area_se, radius_worst, perimeter_worst, area_worst,
                      compactness_worst, concavity_worst, concave_points_worst):
    prediction = model.predict(np.array([[radius_mean, perimeter_mean, area_mean, compactness_mean,
                                          concavity_mean, concave_points_mean, radius_se, perimeter_se,
                                          area_se, radius_worst, perimeter_worst, area_worst,
                                          compactness_worst, concavity_worst, concave_points_worst]]))
    if prediction == 1:
        return "Cancer"
    else:
        return "No Cancer"

if __name__ == '__main__':
    main()
