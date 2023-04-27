import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    data = download_data()
    if option == 'Visualize':
        visualize_data(data)
    elif option == 'Recent Data':
        dataframe(data)
    else:
        predict(data)


@st.cache_resource
def download_data():
    df = yf.download('GLD', start='2008-01-01', end=datetime.date.today(), progress=False)
    return df






scaler = StandardScaler()
model = joblib.load('model.pkl')


def visualize_data(data):
    st.header('The Close Price')
    st.line_chart(data.Close)



def dataframe(data):
    st.header('Recent Data')
    st.dataframe(data.tail(10))



def predict(data):
    df = data[['Close']]

    df['weekly_mean'] = df.Close.rolling(window=7).mean()
    df['monthly_mean'] = df.Close.rolling(window=30).mean()
    df['quarterly_mean'] = df.Close.rolling(window=90).mean()
    df['yearly_mean'] = df.Close.rolling(window=365).mean()

    df = df.dropna()
    # forecast the price

    features = df[['weekly_mean', 'monthly_mean', 'quarterly_mean', 'yearly_mean']].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    df['predicted_gold_price'] = model.predict(features)
    df['signal'] = np.where(df.predicted_gold_price.shift(1) < df.predicted_gold_price,"Buy","No Position")

    prediction = df.tail(1)[['signal','predicted_gold_price']].T
    st.header('Gold Price Prediction')
    st.write("Today's Price")
    st.dataframe(data.Close.tail(1))
    st.write('Next Day Predicted Price')
    st.dataframe(prediction)



if __name__ == '__main__':
    main()
