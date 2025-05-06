import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, jsonify, request
from src.utils import get_data
from src import app
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error

@app.route('/carban_emissions_forcasting', methods=['GET'])
def get_carban_emission_forecast():
    country = request.args.get('country')
    forecast_df = forecast_carban_emissions_forcasting(country)
    return jsonify(forecast_df.to_dict(orient='records'))

@app.route('/carban_emissions_history', methods=['GET'])
def get_carban_emission_history():
    country = request.args.get('country')
    forecast_df = history_carban_emissions_forcasting(country)
    return jsonify(forecast_df.to_dict(orient='records'))

def forecast_carban_emissions_forcasting(country, years_to_forecast=8):
    # Improved forecasting with SARIMA(1,0,0)
    data_selected = get_data()
    country_data = data_selected[data_selected['country'] == country]
    country_data = country_data.set_index('year')
    series = country_data['greenhouse_gas_emissions']
    
    # Data preparation (keeping your original format)
    series = series.reindex(np.arange(2000, 2023))
    series = series.interpolate(method='linear')
    series = series.fillna(method='bfill').fillna(method='ffill')
    series = series[series.index >= 2000]
    series = data_cleaning_func(series)

    # Use SARIMA(1,0,0) directly
    model = SARIMAX(series, order=(1,0,0))
    model_fit = model.fit(disp=False)
    
    # Use get_forecast() instead of forecast() to get confidence intervals
    forecast = model_fit.get_forecast(steps=years_to_forecast)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    # Maintain same output format
    forecasting_years_range = np.arange(country_data.index[-1] + 1, 
                                     country_data.index[-1] + years_to_forecast + 1)
        
    return pd.DataFrame({
        'Year': forecasting_years_range,
        'Carban Emissions Forecasting': forecast_mean.round(3).values,
        'Lower_Bound': conf_int.iloc[:, 0].round(3).values,
        'Upper_Bound': conf_int.iloc[:, 1].round(3).values
    }).reset_index(drop=True)

def history_carban_emissions_forcasting(country, years_to_forecast=8):
    # Maintained exactly as-is for compatibility
    data_selected = get_data()
    country_data = data_selected[data_selected['country'] == country]
    country_data = country_data.set_index('year')
    series = country_data['greenhouse_gas_emissions']

    series = series[series.index >= 2000]
    series = series.reindex(np.arange(2000, 2023))
    series = series.interpolate(method='linear')
    series = series.fillna(method='bfill').fillna(method='ffill')

    return pd.DataFrame({
        'Year': series.index,
        'History_Consumption': series.values
    }).reset_index(drop=True)

def data_cleaning_func(series):
    # Maintained exactly as-is for compatibility
    series = series.drop_duplicates()
    series = series.dropna()
    return series