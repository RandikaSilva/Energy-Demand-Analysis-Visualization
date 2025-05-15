from typing import Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, jsonify, request
from src.utils import get_data
from src import app
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

@app.route('/consumption_forcast', methods=['GET'])
def get_forecast():
    country = request.args.get('country')
    target_column = request.args.get('target_column')
    
    if not country or not target_column:
        return jsonify({"error": "Please provide 'country' and 'target_column' as query parameters"}), 400

    forecast_df = forecast_energy_consumption(country, target_column, 8)
    df = get_data()
    df = load_energy()
    fc_primary = forecast_series(df, target_column, horizon=8)
    return jsonify(fc_primary.to_dict(orient='records'))

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

def load_energy(country: str = "New Zealand", start_year: int = 1965,
                remove_outliers: bool = True) -> pd.DataFrame:
    """Return cleaned annual energy & macro frame for *country* ≥ *start_year*."""

    cols = [
        "country", "year", "primary_energy_consumption",
        "renewables_consumption", "gdp", "population",
    ]

    df = (pd.read_csv('./src/data/World Energy Consumption.csv', usecols=cols)
            .query("country == @country and year >= @start_year")
            .set_index("year")
            .sort_index())

    # Interpolate GDP & population – both directions then ff/bf fill
    for col in ("gdp", "population"):
        df[col] = (df[col]
                    .interpolate(limit_direction="both")
                    .ffill()  # leading
                    .bfill())  # trailing

    # Optional outlier dampening on primary energy
    if remove_outliers:
        y = df["primary_energy_consumption"]
        pct = y.pct_change()
        mask = pct.abs() > 0.25  # >25 % jump year-on-year
        if mask.any():
            df.loc[mask, "primary_energy_consumption"] = np.nan
            df["primary_energy_consumption"] = df["primary_energy_consumption"].interpolate()

    # Exogenous growth rates
    df["gdp_growth"] = df["gdp"].pct_change().fillna(0)
    df["pop_growth"] = df["population"].pct_change().fillna(0)

    # Final sanity-check
    if df.isna().any().any():
        raise ValueError("NaNs remain after cleaning:\n" + df.isna().sum().to_string())

    return df.drop(columns="country")

def fit_arima(y: pd.Series, X: pd.DataFrame):
    return auto_arima(
        np.log1p(y),
        exogenous=X,
        d=1,
        max_p=4,
        max_q=4,
        seasonal=False,
        with_intercept=True,      # drift term
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
    )

def rolling_cv(y: pd.Series, X: pd.DataFrame, splits: int = 5) -> Tuple[float, float]:
    tss = TimeSeriesSplit(n_splits=splits)
    maes, rmses = [], []
    for train_idx, test_idx in tss.split(y):
        m = fit_arima(y.iloc[train_idx], X.iloc[train_idx])
        pred_log = m.predict(n_periods=len(test_idx), exogenous=X.iloc[test_idx])
        pred = np.expm1(pred_log)
        truth = y.iloc[test_idx]
        maes.append(mean_absolute_error(truth, pred))
        rmses.append(np.sqrt(mean_squared_error(truth, pred)))
    return float(np.mean(maes)), float(np.mean(rmses))

def forecast_series(df: pd.DataFrame, target: str,
                    horizon: int = 8,
                    gdp_growth: float = 0.02,
                    pop_growth: float = 0.01) -> pd.DataFrame:

    y = df[target]
    exog_cols = ["gdp_growth", "pop_growth"]
    X = df[exog_cols]

    # Train full model
    model = fit_arima(y, X)

    last = df.iloc[-1]
    future_years = range(df.index[-1] + 1, df.index[-1] + horizon + 1)

    future_X = pd.DataFrame({
        "gdp_growth": np.full(horizon, gdp_growth),
        "pop_growth": np.full(horizon, pop_growth),
    }, index=future_years)

    log_fc, ci = model.predict(n_periods=horizon, exogenous=future_X, return_conf_int=True)
    fc = np.expm1(log_fc)

    out = pd.DataFrame({
        "Year": future_years,
        "Forecast_Consumption": fc.round(2),
        "Lower_Bound": np.expm1(ci[:, 0]).round(2),
        "Upper_Bound": np.expm1(ci[:, 1]).round(2),
    })  

    return out