import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, jsonify, request
from src.utils import get_data
from src import app

@app.route('/demand_vs_generations_v1', methods=['GET'])
def get_demand_vs_generations():
    countries = request.args.get('countries', '')
    countries = [country.strip().strip("'") for country in countries.split(',')] if countries else []

    analized_data = demand_vs_forecasting(countries)
    return jsonify(analized_data.to_dict(orient='records'))

def demand_vs_forecasting(countries):
    data_selected = get_data()
    data_selected = data_selected[data_selected['country'].isin(countries)]
    columns_needed = ['country', 'year', 'electricity_generation', 'electricity_demand']
    data_selected = data_selected[columns_needed]
    
    data_selected = data_cleaning_func(data_selected)

    result = []
    for country in countries:
        subset = data_selected[data_selected['country'] == country]
        subset = subset[subset['electricity_demand'] > 0]
        efficiency = subset['electricity_generation'] / subset['electricity_demand']
        for year, eff in zip(subset['year'], efficiency):
            result.append({'country': country, 'year': year, 'efficiency': eff})
            
    return pd.DataFrame(result)

def data_cleaning_func(dataframe):
    dataframe = dataframe.drop_duplicates()
    dataframe = dataframe.dropna()
    return dataframe