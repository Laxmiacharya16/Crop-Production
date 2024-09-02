#!/usr/bin/env python
# coding: utf-8

# In[11]:


import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model

model_path = "C:/Users/srupa/Downloads/unified mentor/1st project/best_rf_model.pkl"
model = joblib.load(model_path)

# Define all the columns used during training
encoded_columns = [
    'Crop_Year', 'Area', 'State_Andhra Pradesh', 'State_Assam', 'State_Bihar', 'State_Chhattisgarh',
    'State_Gujarat', 'State_Karnataka', 'State_Madhya Pradesh', 'State_Maharashtra', 'State_Odisha',
    'State_Rajasthan', 'State_Tamil Nadu', 'State_Uttar Pradesh', 'State_West Bengal', 'State_other_state',
    'Crop_Arhar/Tur', 'Crop_Gram', 'Crop_Groundnut', 'Crop_Jowar', 'Crop_Maize', 'Crop_Moong(Green Gram)',
    'Crop_Onion', 'Crop_Potato', 'Crop_Rapeseed &Mustard', 'Crop_Rice', 'Crop_Sesamum', 'Crop_Sugarcane',
    'Crop_Urad', 'Crop_Wheat', 'Crop_other_crops', 'Season_Kharif     ', 'Season_Rabi       ',
    'Season_Summer     ', 'Season_Whole Year ', 'Season_Winter     '
]

# Title and description
st.title("Crop Production Prediction App")
st.write("Predict crop production based on area, date, season, crop type, and district.")

# User input widgets
state = st.selectbox("State", [
    'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Gujarat', 'Karnataka',
    'Madhya Pradesh', 'Maharashtra', 'Odisha', 'Rajasthan', 'Tamil Nadu',
    'Uttar Pradesh', 'West Bengal', 'other_state'
])
season = st.selectbox("Season", ['Kharif     ', 'Rabi       ', 'Summer     ', 'Whole Year ', 'Winter     '])
crop = st.selectbox("Crop", [
    'Arhar/Tur', 'Gram', 'Groundnut', 'Jowar', 'Maize', 'Moong(Green Gram)', 'Onion', 'Potato',
    'Rapeseed &Mustard', 'Rice', 'Sesamum', 'Sugarcane', 'Urad', 'Wheat', 'other_crops'
])
area = st.number_input("Area (in hectares)", min_value=0.0, step=0.1)
crop_year = st.number_input("Crop Year", min_value=1900, max_value=2100, step=1)

# Create a DataFrame with all possible one-hot encoded columns initialized to 0
user_input = pd.DataFrame(0, index=[0], columns=encoded_columns)

# Set the selected values to 1 if they exist in the one-hot encoded columns
if f'State_{state}' in user_input.columns:
    user_input[f'State_{state}'] = 1
else:
    user_input['State_other_state'] = 1

if f'Crop_{crop}' in user_input.columns:
    user_input[f'Crop_{crop}'] = 1
else:
    user_input['Crop_other_crops'] = 1

user_input[f'Season_{season}'] = 1

# Add the area and crop year
user_input['Area'] = area
user_input['Crop_Year'] = crop_year

# Ensure all columns are in the correct order expected by the model
user_input = user_input.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict and display the result
if st.button("Predict"):
    prediction = model.predict(user_input)
    st.write(f"Predicted Production: {prediction[0]:.2f}")






# In[ ]:




