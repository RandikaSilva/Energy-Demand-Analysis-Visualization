import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, jsonify, request
from src.utils import get_data
from src import app

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
    data_selected = get_data()
    country_data = data_selected[data_selected['country'] == country]
    country_data = country_data.set_index('year')
    series = country_data['greenhouse_gas_emissions']
    
    # Data cleaning
    # Remove duplicates and NaN values
    series = data_cleaning_func(series)

    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=years_to_forecast)

    forecasting_years_range = np.arange(country_data.index[-1] + 1, country_data.index[-1] + years_to_forecast + 1)
     
    forecast_data = pd.DataFrame({
        'Year': forecasting_years_range,
        'Carban Emissions Forecasting': forecast.values
    })
    
    return forecast_data.reset_index(drop=True)

def history_carban_emissions_forcasting(country, years_to_forecast=8):
    data_selected = get_data()
    country_data = data_selected[data_selected['country'] == country]
    country_data = country_data.set_index('year')
    series = country_data['greenhouse_gas_emissions']

    # Data cleaning
    # Remove duplicates and NaN values
    series = data_cleaning_func(series)

    historical_data = pd.DataFrame({
        'Year': series.index,
        'History_Consumption': series.values
    })

    result = pd.concat([historical_data]).reset_index(drop=True)
    return result

def data_cleaning_func(series):
    series = series.drop_duplicates()
    series = series.dropna()
    return series