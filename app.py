import streamlit as st
from datetime import datetime
import numpy as np
import joblib
import yfinance as yf




model = joblib.load('model.pkl')

def main():

    st.sidebar.header('Gold Price Prediction')
    today = datetime.today()
    start_date = st.sidebar.date_input('Start Date')
    end_date = st.sidebar.date_input('End date', today)
    if st.sidebar.button('Send'):
        if start_date < end_date:
            st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
            model_pred(start_date, end_date)
        else:
            st.sidebar.error('Error: End date must fall after start date')



def model_pred(start, end):
    data = download_data(start, end)
    df = data[['Close']]
    df['weekly_mean'] = df.Close.rolling(window=7).mean()
    df['monthly_mean'] = df.Close.rolling(window=30).mean()
    df['quarterly_mean'] = df.Close.rolling(window=90).mean()
    df['yearly_mean'] = df.Close.rolling(window=365).mean()

    df = df.dropna()
    # forecast the price

    df['predicted_gold_price'] = model.predict(df[['weekly_mean', 'monthly_mean', 'quarterly_mean', 'yearly_mean']])
    df['signal'] = np.where(df.predicted_gold_price.shift(1) < df.predicted_gold_price,"Buy","No Position")

    prediction = df.tail(1)[['signal','predicted_gold_price']].T
    st.header('Gold Price Prediction')
    st.dataframe(prediction)



@st.cache_resource
def download_data(start, end):
    data = yf.download('GLD', start=start, end=end)
    return data


if __name__ == '__main__':
    main()
