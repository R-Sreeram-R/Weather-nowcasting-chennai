import numpy as np 
import pandas as pd 
import streamlit as st
import tensorflow as tf
import sys 
from datetime import datetime, timedelta
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from PIL import Image


def main():
    
        
    st.title("Weather Nowcasting Using Deep Learning 👩‍💻")
    
    image = Image.open("D:\Weather Nowcasting D drive\Chennai_img.jpg")

    
    st.image(image, caption='Chennai', use_column_width=True)   
    
    
    model = st.select_slider('Choose the model for prediction',['Univariate', 'Multi-Variate'])
    
    df = pd.read_csv('D:\Weather Nowcasting D drive\Models\Datasets\Chennai.csv')
    df.index = pd.to_datetime(df['datetime'])
    submit_button = st.button('Predict!')
    
    if model=='Univariate' and submit_button:
        precip = df['precip']    
        precip = precip[-10:]
        WINDOW_SIZE = 5
        data,y = df_to_X_y(precip, WINDOW_SIZE)
        prediction = predict(data,'Univariate')
        st.write(prediction)
        
        
        if prediction < 1.8:
            st.success('It will not rain in the next hour! ☀️')
        else:
            st.error("It will likely rain next hour! 🌧️")
        
    
    elif model == 'Multi-Variate' and submit_button:
        
        data = clean_data(df)        
       
        WINDOW_SIZE = 50                
        data, y =  df_to_X_y2(data,WINDOW_SIZE)   
        
        prediction = predict(data,'Multi-Variate')
        
        st.write(prediction)
        
        if prediction < 5:
            st.success('It will not rain in the next hour! ☀️')
        else:
            st.error("It will likely rain next hour! 🌧️")    
       
        
def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()

    X = []
    y = []

    for i in range(len(df_as_np) - window_size ):
        row = [[a] for a in df_as_np[i:i+window_size]]

        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)

    return np.array(X), np.array(y)


def df_to_X_y2(df, window_size=6):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size][3]
    y.append(label)
  return np.array(X), np.array(y)



def predict(data,model):
    
    if model == 'Univariate':    
        model = load_model('D:\Weather Nowcasting D drive\Models\my_model.h5')
        test_results = model.predict(data).flatten()
        return  test_results[-1]
    else:
        model = load_model('D:\Weather Nowcasting D drive\Models\my_model_multi_20.h5')

        test_results = model.predict(data).flatten()
        
        final_result = test_results[-1]
        
        return inverse_transform(final_result)  
   
    

def inverse_transform(num):
    
    std = 2.4245306735158434
    mean = 0.20557103413654618
    
    original = (num * std) + mean
    
    
    return original   


def clean_data(df):
    
    # not setting index since date time set earlier
    null_pct = df.apply(pd.isnull).sum() / df.shape[0]
    valid_columns = df.columns[null_pct < 0.05]
    
    # hard forcing the values
    valid_columns = ['name', 'datetime', 'temp', 'dew', 'humidity', 'precip',
       'precipprob', 'cloudcover', 'conditions',
        ]
    weather = df[valid_columns].copy()
    numeric_weather = weather.select_dtypes(include=['number']) 
   
    
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(numeric_weather), columns=numeric_weather.columns)
    
    day = 60*60*24
    year = 365.2425*day

    df_standardized['Seconds'] = numeric_weather.index.map(pd.Timestamp.timestamp)


    df_standardized['Day sin'] = np.sin(df_standardized['Seconds'] * (2* np.pi / day))
    df_standardized['Day cos'] = np.cos(df_standardized['Seconds'] * (2 * np.pi / day))
    df_standardized['Year sin'] = np.sin(df_standardized['Seconds'] * (2 * np.pi / year))
    df_standardized['Year cos'] = np.cos(df_standardized['Seconds'] * (2 * np.pi / year))
    df_standardized = df_standardized.drop('Seconds', axis=1)   
   
    
    
    return df_standardized
     
    

    

main()
