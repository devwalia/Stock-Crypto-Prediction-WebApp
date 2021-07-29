# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 17:42:24 2021

@author: HP
"""

#TO run use streamlit run C:\Users\HP\Desktop\StockAnalysis\webapp.py

import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("STOCK & CRYPTO PREDICTION APP")

stocks = ("AAPL","GOOG","RELCAPITAL.NS","PTC.NS","BTC-INR","ETH-INR","ICP1-INR")
selected_stock = st.selectbox("Select Dataset For Prediction",stocks)

n_years = st.slider("Year Of Prediction:",1,4)
period =n_years * 365

#defining function to load the data

@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data....done!!")


#Ploting and Analyzing

st.subheader('Raw Data')
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()


#Forecasting using facebook prophets
df_train = data[{'Date','Close'}]
df_train = df_train.rename(columns= {"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

#ploting forecast datausing plotly

st.write('Forecast Data')

fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)

 #You can now view your Streamlit app in your browser.

  #Local URL: http://localhost:8501
  #Network URL: http://192.168.29.146:8501