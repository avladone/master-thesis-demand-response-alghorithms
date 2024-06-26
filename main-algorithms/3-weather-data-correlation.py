"""
This script loads hourly weather data, merges it with the energy data, and performs correlation analysis.
"""

import pandas as pd
from utilities import plot_scatter, plot_correlation_heatmap

# Load energy dataset
energy_hourly = pd.read_csv('data/outputs/energy_hourly.csv')
energy_hourly['Date'] = pd.to_datetime(energy_hourly['Date'])
energy_hourly.set_index('Date', inplace=True)

# Load weather dataset
weather_data = pd.read_csv('data/inputs/weather_hourly_2023.csv')

# Ensure 'Date' columns are in datetime format
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Ensure all necessary weather columns are present and handle missing values if any
weather_data = weather_data[['AvgTemp_Celsius', 'AvgWindSpeed_kmh', 'SolarRadiation_WM2']].dropna()

# Align Weather Data to Energy Data

# Ensure the weather data has enough rows to match the energy data
if len(weather_data) < len(energy_hourly):
    raise ValueError("Weather data does not have enough rows to match energy data.")

# Create a new DataFrame with weather data aligned to energy data timestamps
aligned_weather_data = weather_data.head(len(energy_hourly)).copy()
aligned_weather_data.index = energy_hourly.index

# Merge Datasets and Correlation Analysis

# Merge datasets on 'Date'
merged_data = energy_hourly.join(aligned_weather_data)

# Filter non-zero solar production for specific analysis
filtered_solar_data = merged_data[merged_data['Solar_MW'] > 0]

# Correlation of weather conditions and energy production
correlation = filtered_solar_data[[
    'AvgTemp_Celsius', 'AvgWindSpeed_kmh', 'Solar_MW', 'Wind_MW', 'SolarRadiation_WM2'
]].corr()

# Visualize Correlations
plot_correlation_heatmap(correlation, 'Correlation Matrix Heatmap')

# Scatter plot for significant correlations
plot_scatter(filtered_solar_data, 'SolarRadiation_WM2', 'Solar_MW', 
             'Solar Radiation vs. Solar Energy Production', 
             'Solar Radiation (W/m²)', 'Solar Energy Production (MW)')

plot_scatter(filtered_solar_data, 'AvgWindSpeed_kmh', 'Wind_MW', 
             'Wind Speed vs. Wind Energy Production', 
             'Wind Speed (km/h)', 'Wind Energy Production (MW)')

# Save the merged dataset
merged_data.to_csv('data/outputs/merged_energy_weather_data.csv')

# Provide the path to the saved file
print("Merged dataset saved to: data/outputs/merged_energy_weather_data.csv")