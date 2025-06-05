# Energy Demand Analysis & Visualization - New Zealand 🇳🇿

This project provides a comprehensive analysis and forecasting of New Zealand’s energy consumption, carbon emissions, and renewable energy share using **Adaptive Business Intelligence (ABI)** techniques. It integrates **ARIMA/SARIMA** time series models, **Power BI dashboards**, and a **Flask API** to deliver insights and real-time energy forecasting tools.

## 📊 Project Overview

New Zealand aims to achieve **net zero carbon emissions by 2050**. This project supports that goal by:
- Forecasting carbon emissions and primary energy consumption
- Evaluating demand vs. generation efficiency
- Visualizing renewable energy share trends

The solution is built using:
- Python (Jupyter Notebooks)
- ARIMA/SARIMA time series models
- Power BI for dashboards
- Flask for API integration

---

## 📈 Power BI Dashboards

Interactive dashboards were built to display:
- Historical and forecasted carbon emissions
- Renewable energy contributions over time
- Demand vs. generation efficiency analysis

Power BI is connected to the backend via **Flask API** for dynamic updates.

### 🔹 Carbon Emissions Forecast
![Carbon Emissions Forecast](images/carbon_forecast.png)

### 🔹 Primary Energy Consumption Forecast
![Energy Consumption Forecast](images/energy_consumption.png)

### 🔹 Renewable Energy Share
![Renewable Energy Share](images/renewable_share.png)

### 🔹 Demand vs Generation Efficiency
![Demand vs Generation](images/demand_vs_generation.png)

---

## 📂 Repository Structure

```bash
Energy-Demand-Analysis-Visualization/
│
├── notebooks/                  # Jupyter Notebooks for data cleaning, EDA, modeling
│   └── energy_forecasting.ipynb
│
├── api/                        # Flask app for API integration with Power BI
│   └── app.py
│
├── data/                       # Sample input datasets
│   └── world_energy_consumption.csv
│
├── dashboard/                  # Power BI dashboard files
│   └── energy_dashboard.pbix
│
├── models/                     # Serialized model files (if any)
│
├── README.md                   # This file
└── requirements.txt            # Required Python packages
