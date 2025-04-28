from flask import Flask

app = Flask(__name__)

from src.controller import consumption_forecasting