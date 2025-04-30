import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, jsonify, request
from src.utils import get_data
from src import app

@app.route('/renewable_energy_growth_tracking_plot_V1', methods=['GET'])
def get_renewable_energy_growth_tracking_plot():
    country = request.args.get('country')

    analized_data = renewable_energy_growth_tracking_plot(country)
    return jsonify(analized_data.to_dict(orient='records'))

@app.route('/renewable_energy_growth_tracking_insights', methods=['GET'])
def get_renewable_energy_growth_tracking_insights():
    country = request.args.get('country')

    analized_data = generate_insights(country)
    return jsonify(analized_data.to_dict(orient='records'))

def renewable_energy_growth_tracking_plot(country):
    data_selected = get_data()
    data_selected = data_selected[(data_selected['country'] == country) & (data_selected['year'] >= 1989)]
    renewable_columns  = [
    'biofuel_share_elec', 'hydro_share_elec', 'solar_share_elec', 'wind_share_elec', 'other_renewables_share_elec']
    
    data_selected = data_selected[['year'] + renewable_columns]
    data_selected = data_selected.fillna(0)
    data_selected = data_cleaning_func(data_selected)
    
    result = []
    for _, row in data_selected.iterrows():
        for col in renewable_columns:
            source = col.replace('_share_energy', '').capitalize()
            result.append({
                "Year": row['year'],
                "Source": source,
                "Values": row[col]
            })

    return pd.DataFrame(result)

def generate_insights(country):
    data_selected = get_data()
    data_selected = data_selected[(data_selected['country'] == country) & (data_selected['year'] >= 1989)]
    renewable_columns  = [
    'biofuel_share_elec', 'hydro_share_elec', 'solar_share_elec', 'wind_share_elec', 'other_renewables_share_elec']
    
    data_selected = data_selected[['year'] + renewable_columns]
    
    data_selected = data_selected.fillna(0)
    data_selected = data_cleaning_func(data_selected)
    
    insights = []
    latest_year = data_selected['year'].max()
    earliest_year = data_selected['year'].min()

    latest_data = data_selected[data_selected['year'] == latest_year]
    earliest_data = data_selected[data_selected['year'] == earliest_year]

    for col in renewable_columns:
        source = col.replace('_share_energy', '').capitalize()
        start = earliest_data[col].values[0]
        end = latest_data[col].values[0]
        change = end - start
        direction = "increased" if change > 0 else "decreased"
        insights.append({
            "Insights": f"The share of energy from {source} has {direction} from {round(start, 2)}% in {earliest_year} to {round(end, 2)}% in {latest_year}, "
                       f"with a net change of {round(change, 2)}%."
        })

    return pd.DataFrame(insights)

def data_cleaning_func(dataframe):
    dataframe = dataframe.drop_duplicates()
    dataframe = dataframe.dropna()
    return dataframe
