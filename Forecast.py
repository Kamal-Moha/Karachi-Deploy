"""
Freely adapted from another Omdena app:
https://saudi-arabia-industrial-co2.streamlit.app
"""

# Import of all required libraries
import requests
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import statsmodels.api as sm
import darts
import random
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
from darts.models import RegressionModel, CatBoostModel, RandomForest, LightGBMModel, XGBModel, RNNModel
from darts.metrics import rmse, mape
from statsmodels.tsa.arima.model import ARIMA
import os
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Set the work directory to retrieve all data
json_data = {
    "data_columns" : "weathercode,temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,sunrise,sunset,shortwave_radiation_sum,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,et0_fao_evapotranspiration",
    "evo_model" : "evapotranspiration_model.pt",
    "pre_rate_model" : "precipitation_rate_model.pt"
}

# Load evo_model
# evo_model = RNNModel.load(open(r"evapotranspiration_model.pt", 'rb', encoding='utf-8'), map_location='cpu')
evo_model = RNNModel.load("evapotranspiration_model.pt", map_location='cpu')
evo_model.to_cpu()

# Load pre_rate_model
# pre_rate_model = RNNModel.load(open(r'precipitation_rate_model.pt', 'rb', encoding='utf-8'), map_location='cpu')
pre_rate_model = RNNModel.load("precipitation_rate_model.pt", map_location='cpu')
pre_rate_model.to_cpu()

# Format to datetime
data_columns = json_data['data_columns']
now = datetime.now() - relativedelta(days=7)
start = now - relativedelta(months=11)
date_string_end = now.strftime('%Y-%m-%d')
date_string_start = start.strftime('%Y-%m-%d')
date_pred = []
for date in pd.date_range(start=datetime.now() - relativedelta(days=6), periods=10):
    date_pred.append(date.strftime('%Y-%m-%d'))

# Plug to live API to retrive live data
url = "https://archive-api.open-meteo.com/v1/archive"
cities = [
    { "name": "Karachi", "country": "Pakistan", "latitude": 24.8608, "longitude": 67.0104 }
]
cities_df =[]
for city in cities:
    params = {"latitude":city["latitude"],
            "longitude":city['longitude'],
            "start_date": date_string_start,
            "end_date": date_string_end,
            "daily": data_columns,
            "timezone": "GMT",
            "min": date_string_start,
            "max": date_string_end,
    }
    res = requests.get(url, params=params)
    data = res.json()
    df = pd.DataFrame(data["daily"])
    df["latitude"] = data["latitude"]
    df["longitude"] = data["longitude"]
    df["elevation"] = data["elevation"]
    df["country"] = city["country"]
    df["city"] = city["name"]
    cities_df.append(df)
concat_df = pd.concat(cities_df, ignore_index=True)
concat_df.set_index('time', inplace=True)
print(concat_df.columns)
total_hours = concat_df['precipitation_hours'].sum()
concat_df['precipitation_rate'] = concat_df['precipitation_sum']/total_hours

# generate prediction for max temp
max_temp = concat_df['temperature_2m_max'].values
# Define and fit the model
max_temp_model = ARIMA(max_temp, order=(5,1,0))
max_temp_model = max_temp_model.fit()
# Make predictions
max_temp_predictions = max_temp_model.predict(start=1, end=10)

# generate prediction for evo_transpiration
mean_evo = concat_df['et0_fao_evapotranspiration'].mean()
et0_fao_evapotranspiration = TimeSeries.from_series(concat_df['et0_fao_evapotranspiration'].values, fillna_value=mean_evo)
scaler = StandardScaler()
transformer = Scaler(scaler)
series_transformed = transformer.fit_transform(et0_fao_evapotranspiration)
st.write(evo_model)
