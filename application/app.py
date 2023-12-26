import numpy as np 
import pandas as pd 
import streamlit as st
import tensorflow as tf
import sys 
from datetime import datetime, timedelta
import requests
from tensorflow.keras.models import load_model

#######################


api_key = 'H2YL8JVLF5RUQD7SY4WZ8CYN4'
api_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"




location = "City,Country"  # Replace with the desired location
start_date = "2023-01-01"



def main():
   
    
   st.title("Weather Nowcasting Application!🏖️")

   model = st.select_slider('Choose the model for prediction',['Univariate', 'Multi-Variate'])

   selected_date = st.date_input("Select a starting date", datetime.today())
   # Print the selected date
   st.write("Selected date:", selected_date)   
   
   location = st.text_input('Input the city in the format -> "City,Country"',)   
   url = f"{api_url}/{location}/{start_date}/{selected_date}?key={api_key}&unitGroup=metric&dayStartTime=0:00:00&dayEndTime=23:59:59&contentType=json"   
   response = requests.get(url)

   if response.status_code==200:
      json_data = response.json()

      hourly_data = pd.DataFrame(json_data['days'][0]['hours'])

      hourly_data = hourly_data['precip']
       
      WINDOW_SIZE = 5
      X, y = df_to_X_y(hourly_data, WINDOW_SIZE)

      predictions = predict(X)



      print(predictions)

   else:
      print(f"Error: {response.status_code}")
      print(response.text)

   
       

   
    








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
    label = df_as_np[i+window_size][0]
    y.append(label)
  return np.array(X), np.array(y)


def load_model():
   ...


def predict(data):
   model = load_model('Models\my_model.h5')

   test_results = model.predict(data).flatten()
   
   
   return  test_results.iloc[-1]




main()




