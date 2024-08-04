#import libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.categorical_encoder import encode_categorical
from src.preprocessing_pipeline import create_preprocessing_pipeline

#fucntion to load model
def load_model():
    lgbm_model = joblib.load('models/lgbm_model.joblib')
    dec_tree = joblib.load('models/dec_tree.joblib')

    return lgbm_model, dec_tree

#function to load and fit preprocessor
def load_and_fit_preprocessor(file_path):
    flight_df = pd.read_csv(file_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(flight_df)

    return preprocessor 

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
def make_prediction(model, preprocessor, input_df):
    input_df_transformed = preprocessor.transform(encode_categorical(input_df))
    prediction = model.predict(input_df_transformed)

    return 'Satisfied 😊' if prediction[0] else 'Not Satisfied 😞'

#display logo and hero image
st.sidebar.image('images/Image20240804233637.png', width=100)
st.image('images/Image20240804233649.jpg', use_column_width=True)

st.title('Cebu Pacific Customer Satisfaction Prediction ✈️')
st.header('Please Tell Us About Your Experience! 😄')


#load models and preprocessor 
lgbm_model, dec_tree = load_model()
preprocessor = load_and_fit_preprocessor('data/transformed/_new_flight_df.csv')

st.write("""
         This app predicts flight satisfaction based on user inputs. 
         You can select between 2 machine learning models: LightGBM and Decision Tree
         """)


#collect user input
st.sidebar.header('User Input')
input_df = user_input_features()

#display user inputs
st.write('### User Inputs')

#iterate over each items in the usert inputs

for key, value in input_df.iloc[0].items():
    st.write(f'**{key}**: {value}')

#model selection 
model_choice = st.sidebar.selectbox('Choose Model 🤖', ['LightGBM', 'Decision Tree'])
selected_model = lgbm_model if model_choice == 'LightGBM' else dec_tree
st.write(f'### Selected model: {model_choice}')

#prediction button
if st.button('Predict'):
    try: 
        result = make_prediction(selected_model, preprocessor, input_df)
        st.success(f'You are {result}')

    except Exception as e:
        st.error(f'Error Occured during prediction. {e}\n Please contact support')