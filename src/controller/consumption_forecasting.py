import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, jsonify, request
from src.utils import get_data
from src import app

@app.route('/consumption_forcast', methods=['GET'])
def get_forecast():
    country = request.args.get('country')
    target_column = request.args.get('target_column')
    
    if not country or not target_column:
        return jsonify({"error": "Please provide 'country' and 'target_column' as query parameters"}), 400

    forecast_df = forecast_energy_consumption(country, target_column)
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
    if not series.index.equals(exog_vars.index):
        raise ValueError("The indices for endog (series) and exog (exog_vars) are not aligned.")

    model = SARIMAX(series, exog=exog_vars, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    last_exog = exog_vars.iloc[-1]
    future_exog = pd.DataFrame([last_exog] * years_to_forecast, columns=exog_vars.columns)
    
    forecast = model_fit.forecast(steps=years_to_forecast, exog=future_exog)
    forecast_years = np.arange(country_data.index[-1] + 1, country_data.index[-1] + years_to_forecast + 1)
     
    forecast_data = pd.DataFrame({
        'Year': forecast_years,
        'Forecast_Consumption': forecast.round(3).values
    })

    result = pd.concat([forecast_data]).reset_index(drop=True)
    return result

def data_cleaning_func(series):
    series = series.drop_duplicates()
    series = series.dropna()
    return series

def history_data_energy_consumption(country, target_column, years_to_forecast=8):
    data_selected = get_data()
    country_data = data_selected[data_selected['country'] == country]
    country_data = country_data.set_index('year')
    series = country_data[target_column]

    # Data cleaning
    # Remove duplicates and NaN values
    series = data_cleaning_func(series)

    historical_data = pd.DataFrame({
        'Year': series.index,
        'History_Consumption': series.values
    })

    result = pd.concat([historical_data]).reset_index(drop=True)
    return result

def projected_energy_consumption(countries, year = 8):
    forecast_summary = []
    
    for country in countries:
        forecast_results = {}
        primary_consumption = forecast_energy_consumption(country, target_column='primary_energy_consumption', years_to_forecast=year)
        renewable_consumption = forecast_energy_consumption(country, target_column='renewables_consumption', years_to_forecast=year)
        forecast_results[country] = {
                'total_energy_forecast': primary_consumption,
                'renewable_energy_forecast': renewable_consumption
                }
        total_forecast = forecast_results[country]['total_energy_forecast']
        renewables_forecast = forecast_results[country]['renewable_energy_forecast']
        forecast_summary.append({
            'Country': country,
            '2030 Total Energy (TWh)': total_forecast.iloc[-1].item(),
            '2030 Renewables (TWh)': renewables_forecast.iloc[-1].item(),
            'Renewables Share (%)': ((renewables_forecast.iloc[-1] / total_forecast.iloc[-1]) * 100).item()})
    return pd.DataFrame(forecast_summary)