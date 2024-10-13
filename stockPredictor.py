import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title = ("Stock Prediction App")

stock = st.text_input("Enter the name of the stock: ")
n_years = st.slider("Years for prediction: ", 1 , 10)

period = n_years * 365

if (not stock):
    st.info('The program will run once you have entered a stock.')
    st.stop()
    
@st.cache_resource   
def load_data(value):
    data = yf.download(value, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Download Data...")

data = load_data(stock)
data_load_state.text("Data Uploaded")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y= data['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y= data['Close'], name = 'stock_close'))
    fig.layout.update(title_text="Past to Current Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']]

df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods = period)

forecast = model.predict(future)

st.subheader('Future prediction')
st.write(forecast.tail())

fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = model.plot_components(forecast)
st.write(fig2)






