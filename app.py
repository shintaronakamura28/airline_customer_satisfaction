#import libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.categorical_encoder import encode_categorical
from src.preprocessing_pipeline import create_preprocessing_pipeline
from lime.lime_tabular import LimeTabularExplainer
import shap
import lime
import config

def st_shap(plot):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html)

@st.cache_resource
#fucntion to load model
def load_model():
    lgbm_model = joblib.load('models/lgbm_model.joblib')
    dec_tree = joblib.load('models/dec_tree.joblib')

    return lgbm_model, dec_tree

@st.cache_data
#function to load and fit preprocessor
def load_and_fit_preprocessor(file_path):
    flight_df = pd.read_csv(file_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(flight_df)

    return preprocessor 

def load_models():
    lgbm_model = joblib.load(config.MODEL_LGBM_PATH)
    dec_tree = joblib.load(config.MODEL_DEC_TREE_PATH)
    return lgbm_model, dec_tree

#fucntion to collect user input
def user_input_features():
    inputs = {
        'Online boarding': st.sidebar.slider('Online boarding', 1 , 5, 3),
        'Inflight entertainment': st.sidebar.slider('Inflight entertainment', 1 ,5, 3),
        'Checkin service': st.sidebar.slider('Checkin service', 1 , 5, 3),
        'Seat comfort': st.sidebar.slider('Seat comfort', 1 ,5, 3),
        'Inflight wifi service': st.sidebar.slider('Inflight WIFI service', 1 , 5, 3),
        'Baggage handling': st.sidebar.slider('Baggage handling', 1 , 5, 3),
        'Flight Distance': st.sidebar.number_input('Flight Distance', 0, 10000, 100),
        'Business Travel': st.sidebar.selectbox( 'Business Travel', ['Yes', 'No']),
        'Loyal Customer': st.sidebar.selectbox('Loyal Customer', ['Yes', 'No']),
        'Age': st.sidebar.number_input('Age', 0 , 100, 18),
    }

    return pd.DataFrame(inputs, index=[0])

#function to make predictions 
def make_prediction(model, preprocessor, input_df, explainer, lime_explainer):
    try:
        input_df_transformed = preprocessor.transform(encode_categorical(input_df))
        prediction = model.predict(input_df_transformed)

        shap_values = explainer.shap_values(input_df_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] # assuming binary classifcation,  use the second class
        
        lime_explanation = lime_explainer.explain_instance(
            data_row=input_df_transformed[0],
            predict_fn=model.predict_proba,
            num_features=config.LIME_NUM_FEATURES
        )
        
        return 'Satisfied 😊' if prediction[0] else 'Not Satisfied 😞'
    
    except Exception as e:
        st.error(f"Error occured during prediction {str(e)}. Please contact support")
        return None, None, None, None, None

#display logo and hero image
st.sidebar.image('images/Image20240804233637.png', width=100)
st.image('images/Image20240804233649.jpg', use_column_width=True)

st.title('Cebu Pacific Customer Satisfaction Prediction ✈️')
st.header('Please Tell Us About Your Experience! 😄')

#load models and preprocessor 
lgbm_model, dec_tree = load_model()
preprocessor = load_and_fit_preprocessor(config.DATA_PATH)

model_choice = st.sidebar.selectbox('Choose Model 🛠️', ['LightGBM', 'Decision Tree'])
selected_model = lgbm_model if model_choice == 'LightGBM' else dec_tree

explainer = shap.TreeExplainer(selected_model)
lime_explainer = LimeTabularExplainer(
    training_data_preprocessor.transform(encode_categorical(flight_df)),
    class_names=['Not Satsified', 'Satisfied'],
    mode='Classification'
)

st.write("""
         This app predicts flight satisfaction based on user inputs. 
         You can select between 2 machine learning models: LightGBM and Decision Tree
         """)

#collect user input
st.sidebar.header('User Input')
input_df = user_input_features()

#display user inputs
st.write('### User Inputs')
for key, value in input_df.iloc[0].items():
    st.write(f'**{key}**: {value}')


#prediction button
if st.button('Make Prediction ✨'):
    result, shap_values, expected_value, input_df_transformed, lime_explanation = make_prediction(
        selected_model, preprocessor, input_df, explainer, lime_explainer
    )
    if result:
        st.write(f'### Prediction: {result}')
        
        # Display SHAP force plot
        st.write('### SHAP Force Plot:')
        st_shap(shap.force_plot(expected_value, shap_values[0], input_df_transformed[0]))
        
        # Display LIME explanation
        st.write('### LIME Explanation:')
        lime_html = lime_explanation.as_html()
        st.components.v1.html(lime_html)