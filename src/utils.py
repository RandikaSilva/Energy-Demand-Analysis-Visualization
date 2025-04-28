import pandas as pd

def get_data():
    # Load and filter the data
    main_df = pd.read_csv('./src/data/World Energy Consumption.csv')
    return main_df
