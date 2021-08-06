#Importing all the library
from urllib.parse import quote
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
plt.style.use('fivethirtyeight')
from datetime import date
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import streamlit as st


#Loading the model
model = load_model('stock_prediction_model.h5')
start = '2000-01-01'
end = '2021-07-31'

#Function for Visualisation
def visualisation(q):
 fig = plt.figure(figsize=(12 , 6))
 plt.plot(q.Close , linewidth = 3)
 plt.xlabel("Year")
 plt.ylabel("Price")
 plt.legend(["Closing Price"], loc ="lower right")
 st.pyplot(fig)

def visualisation2(q): 
 ma100 = q.Close.rolling(100).mean()
 fig = plt.figure(figsize=(12 , 6))
 plt.plot(ma100 , linewidth = 3)
 plt.plot(q.Close , linewidth = 3)
 plt.xlabel("Year")
 plt.ylabel("Price")
 plt.legend(["Closing Price", "Moivng Average 100"], loc ="lower right")
 st.pyplot(fig)

def visualisation3(q):
 ma100 = q.Close.rolling(100).mean()
 ma200 = q.Close.rolling(200).mean()
 fig2 = plt.figure(figsize=(12 , 6))
 plt.plot(q.Close , linewidth = 3 , color= 'blue')
 plt.plot(ma100 , linewidth = 3 , color = 'red')
 plt.plot(ma200 , linewidth = 3 , color = 'green')
 plt.xlabel("Year")
 plt.ylabel("Price")
 plt.legend(["Closing Price" , "Moving Average 100", "Moving Average 200"], loc ="lower right")
 st.pyplot(fig2)

def visualisation4(q):
 data_training = pd.DataFrame(q['Close'][0:int(len(q)*0.70)])
 data_testing = pd.DataFrame(q['Close'][int(len(q)*0.70):int(len(q))])   
 scaler = MinMaxScaler(feature_range=(0,1))
 data_training_array = scaler.fit_transform(data_training)
 past_100_days = data_training.tail(100) #Last 100 days of data from training dataset
 final_q = past_100_days.append(data_testing , ignore_index=True)
 input_data = scaler.fit_transform(final_q) #Scaling down the testing data in the range 0 and 1
 x_test = []
 y_test = []
 for i in range(100,input_data.shape[0]):
   x_test.append(input_data[i-100 : i])
   y_test.append(input_data[i , 0])
 x_test , y_test = np.array(x_test) , np.array(y_test)
 y_predicted = model.predict(x_test)
 scaler = scaler.scale_
 scale_factor = 1/scaler
 y_predicted = y_predicted * scale_factor
 y_test = y_test * scale_factor

 fig3 = plt.figure(figsize = (12 , 6))
 plt.plot(y_test , 'blue' , label = 'Original Price')
 plt.plot(y_predicted , 'red' , label = 'Predicted Price')
 plt.xlabel("Time")
 plt.ylabel("Price")
 plt.legend(['Original Price', 'Predicted Price'], loc ="lower right")
 st.pyplot(fig3)

#Start
st.title('Stock Prediction App')
user_input = st.text_input('Enter the stock ticker for which you wanna predict', 'AAPL')
quote = yf.download(user_input, start, end)

#Describing the Data
st.header('Description of Data from 2000 - 2021')
st.write(quote.describe())

#Choosing the date
st.header("Live Data")
dt_start = date.today().strftime("%Y-%m-%d") #Current date
dt_end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
live = yf.download(user_input,start=dt_start, end=dt_end)
st.write(" Today's Open : " , live[['Open']].to_numpy())
st.write(" Today's Close : " , live[['Close']].to_numpy())

#Main 
try:
 date_str_start = st.text_input("Please enter the start day ", "2021-06-09")
 date_str_end = st.text_input("Please enter the end day" , "2021-06-10")
 #Actual quote of stock
 quote2 = yf.download(user_input,start=date_str_start, end=date_str_end)
 actual_quote = quote2[['Close']].to_numpy()
 actual_quote[0][0] = np.format_float_positional(actual_quote[0][0], precision=3)
 actual_open = quote2[['Open']].to_numpy()
 actual_open[0][0] = np.format_float_positional(actual_open[0][0], precision=3)
 #Getting the predicted quote
 quote_p = yf.download(user_input,start=date_str_start, end=date_str_end)
 new_quote_p = quote_p.filter(['Close']) 
 last_100_days = new_quote_p[-100:].values
 scalar = MinMaxScaler(feature_range=(0,1))
 last_100_days_scaled = scalar.fit_transform(last_100_days)
 X_test = []
 X_test.append(last_100_days_scaled)
 X_test = np.array(X_test)
 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
 pred_price = model.predict(X_test)
 pred_price = scalar.inverse_transform(pred_price)
 predicted = pred_price
 predicted[0][0] = np.format_float_positional(predicted[0][0], precision=2)

 #Outputting the stock prices
 st.header("Price of Stock ")
 st.write('Opening Price of the Stock : ' ,actual_open[0][0])
 st.write('Actual Closing Price of the Stock : ' , actual_quote[0][0])
 st.write('Predicted Closing Price of the Stock : ' , predicted[0][0])

 #Visualisation
 st.header("Visualisations")
 st.subheader('Closing Price vs Time')
 visualisation(quote)
 st.subheader('Closing Price vs Moving Average of 100 days')
 visualisation2(quote)
 st.subheader('Closing Price vs Moving Average of 100 days vs Moving Average of 200 days')
 visualisation3(quote)
 st.subheader('Predicted Price vs Original Price')
 visualisation4(quote)

except ValueError:
 st.error("Please enter a valid date.")
