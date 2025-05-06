import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, jsonify, request
from src.utils import get_data
from src import app
import warnings
warnings.filterwarnings("ignore")

@app.route('/consumption_forcast', methods=['GET'])
def get_forecast():
    country = request.args.get('country')
    target_column = request.args.get('target_column')
    
    if not country or not target_column:
        return jsonify({"error": "Please provide 'country' and 'target_column' as query parameters"}), 400

    forecast_df = forecast_energy_consumption(country, target_column, 8)
    return jsonify(forecast_df.to_dict(orient='records'))

@app.route('/consumption_history', methods=['GET'])
def get_history():
    country = request.args.get('country')
    target_column = request.args.get('target_column')
    
    if not country or not target_column:
        return jsonify({"error": "Please provide 'country' and 'target_column' as query parameters"}), 400

    forecast_df = history_data_energy_consumption(country, target_column)
    return jsonify(forecast_df.to_dict(orient='records'))

@app.route('/projected_energy_consumption', methods=['GET'])
def get_projected_energy_consumption():
    countries = request.args.get('country')
    if countries:
        countries = [country.strip() for country in countries.split(',')]
    
    if countries is None:
        return jsonify({"error": "Please provide 'country' and 'target_column' as query parameters"}), 400

    forecast_df = projected_energy_consumption(countries, year=8)
    return jsonify(forecast_df.to_dict(orient='records'))

def forecast_energy_consumption(country, target_column, years_to_forecast=8):
    data_selected = get_data()
    country_data = data_selected[data_selected['country'] == country]
    country_data = country_data.set_index('year')
    series = country_data[target_column]
    exog_vars = country_data[['gdp', 'population']].fillna(method='ffill').fillna(method='bfill')
    series = data_cleaning_func(series)
    exog_vars = exog_vars.loc[series.index]

    # Model selection based on target_column
    if target_column == 'primary_energy_consumption':
        order = (2, 1, 2)  # Best for primary energy
    else:
        order = (1, 1, 0)  # Best for renewables

    model = SARIMAX(series, exog=exog_vars, order=order, seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)

    last_exog = exog_vars.iloc[-1]
    future_exog = pd.DataFrame([last_exog] * years_to_forecast, 
                             columns=exog_vars.columns)
    
    forecast = model_fit.get_forecast(steps=years_to_forecast, exog=future_exog)
    forecast_years = list(range(country_data.index[-1] + 1, 
                            country_data.index[-1] + years_to_forecast + 1))
    
    # Get confidence intervals
    conf_int = forecast.conf_int()
    
    return pd.DataFrame({
        'Year': forecast_years,
        'Forecast_Consumption': forecast.predicted_mean.round(3).values,
        'Lower_Bound': conf_int.iloc[:, 0].round(3).values,
        'Upper_Bound': conf_int.iloc[:, 1].round(3).values
    }).reset_index(drop=True)

def data_cleaning_func(series):
    """Clean the time series data"""
    series = series.drop_duplicates()
    series = series.dropna()
    return series

def history_data_energy_consumption(country, target_column, years_to_forecast=8):
    """Get historical data (unchanged from original)"""
    data_selected = get_data()
    country_data = data_selected[data_selected['country'] == country]
    country_data = country_data.set_index('year')
    series = country_data[target_column]
    series = data_cleaning_func(series)

    historical_data = pd.DataFrame({
        'Year': series.index,
        'History_Consumption': series.values
    })

    return historical_data.reset_index(drop=True)

def projected_energy_consumption(countries, year=8):
    """Generate energy projections (unchanged from original)"""
    forecast_summary = []
    
    for country in countries:
        forecast_results = {}
        primary_consumption = forecast_energy_consumption(country, 
                                                         target_column='primary_energy_consumption', 
                                                         years_to_forecast=year)
        renewable_consumption = forecast_energy_consumption(country, 
                                                          target_column='renewables_consumption', 
                                                          years_to_forecast=year)
        
        forecast_summary.append({
            'Country': country,
            '2030 Total Energy (TWh)': primary_consumption.iloc[-1]['Forecast_Consumption'],
            '2030 Renewables (TWh)': renewable_consumption.iloc[-1]['Forecast_Consumption'],
            'Renewables Share (%)': ((renewable_consumption.iloc[-1]['Forecast_Consumption'] / 
                                    primary_consumption.iloc[-1]['Forecast_Consumption']) * 100)
        })
    
    return pd.DataFrame(forecast_summary)