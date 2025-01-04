import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title = ("Stock Track")

st.markdown("# Stock Track")

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

data_load_state = st.text("Downloading Data...")

data = load_data(stock)
data_load_state.text("Data Uploaded")



def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y= data['Close'], name = 'stock_close'))
    fig.layout.update(title_text="Past Prices", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']]

df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods = period)

forecast = model.predict(future)

st.subheader('Future prediction')

fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)



st.subheader('Compare Stocks Past Trends')


stock1 = st.text_input("Enter the name of the first stock: ")
stock2 = st.text_input("Enter the name of the second stock: ")


if (not stock1 or not stock2):
    st.info('The program will run once you have entered stocks.')
    st.stop()


data_load_state = st.text("Downloading Data...")

data1 = load_data(stock1)
data2 = load_data(stock2)
data_load_state.text("Data Uploaded")


fig = go.Figure()
def plot_compare_data():
    fig.add_trace(go.Scatter(x=data1['Date'], y= data1['Close'], name = stock1))
    fig.add_trace(go.Scatter(x=data2['Date'], y= data2['Close'], name = stock2))
    fig.layout.update(title_text="Past Prices", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_compare_data()


st.markdown("---")

st.markdown(
    "https://github.com/coderSasidharan/StockDataPredictor"
)
