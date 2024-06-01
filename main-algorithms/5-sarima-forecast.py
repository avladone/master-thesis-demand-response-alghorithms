"""
This script performs SARIMA forecasts on daily energy demand and production data.
"""

import pandas as pd
from utilities import sarima_forecast

# Load the energy dataset with daily resampled data
energy_data = pd.read_csv('data/outputs/energy_daily.csv')
energy_data['Date'] = pd.to_datetime(energy_data['Date'])
energy_data.set_index('Date', inplace=True)

# List of production types to forecast
production_types = {
    'Demand_MW': 'Energy Demand',
    'Solar_MW': 'Solar Energy Production',
    'Wind_MW': 'Wind Energy Production',
    'Coal_MW': 'Coal Energy Production',
    'Hydrocarbons_MW': 'Hydrocarbons Energy Production',
    'Water_MW': 'Water Energy Production',
    'Nuclear_MW': 'Nuclear Energy Production',
    'Biomass_MW': 'Biomass Energy Production',
    'Import_Positive_MW': 'Imported Energy',
    'Export_Positive_MW': 'Exported Energy'
}

# Dictionary to store forecast dataframes
forecast_dfs = {}

# Perform SARIMA forecasts for each production type and store results
for column, title in production_types.items():
    forecast_df = sarima_forecast(
        energy_data, 
        column, 
        title, 
        order=(2, 1, 2), 
        seasonal_order=(1, 1, 1, 7), 
        steps=365
    )
    forecast_dfs[column] = forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']]

# Combine all forecast dataframes into a single dataframe
combined_forecast_df = pd.concat(forecast_dfs, axis=1)
combined_forecast_df.columns = [f'{col}_{metric}' for col in production_types.keys() for metric in ['mean', 'mean_ci_lower', 'mean_ci_upper']]

# Save the combined forecast dataframe to a CSV file
combined_forecast_df.to_csv('data/outputs/combined_sarima_forecast.csv')

print("Combined SARIMA forecast saved to data/outputs/combined_sarima_forecast.csv")