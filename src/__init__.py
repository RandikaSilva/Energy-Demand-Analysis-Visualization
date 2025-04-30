from flask import Flask

app = Flask(__name__)

from src.controller import consumption_forecasting, carban_emissions_forcasting, demand_vs_generation