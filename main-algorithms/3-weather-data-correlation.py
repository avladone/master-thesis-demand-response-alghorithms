"""
This script loads hourly weather data, merges it with the energy data, and performs correlation analysis.
"""

# Step 1: Import Libraries and Load Datasets

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load energy dataset
energy_hourly = pd.read_csv('energy_hourly.csv')
energy_hourly['Date'] = pd.to_datetime(energy_hourly['Date'])
energy_hourly.set_index('Date', inplace=True)

# Load weather dataset
weather_data = pd.read_csv('weather_hourly.csv')

# Ensure 'Date' columns are in datetime format and set as index
if 'Date' in weather_data.columns:
    weather_data['Date'] = pd.to_datetime(weather_data['Date'])
    weather_data.set_index('Date', inplace=True)
else:
    raise KeyError("The 'Date' column is missing from the weather data.")

# Ensure all necessary weather columns are present and handle missing values if any
weather_data = weather_data[['AvgTemp_Celsius', 'AvgWindSpeed_kmh', 'SolarRadiation_WM2']].dropna()

# Step 3: Merge Datasets and Correlation Analysis

# Reset index for merging
energy_hourly.reset_index(inplace=True)

# Merge datasets on 'Date'
merged_data = pd.merge(energy_hourly, weather_data, on='Date')

# # Filter non-zero solar production
# merged_data = merged_data[merged_data['Solar_MW'] > 0]

# Correlation of weather conditions and energy production
correlation = merged_data[[
    'AvgTemp_Celsius', 'AvgWindSpeed_kmh', 'Solar_MW', 'Wind_MW', 'SolarRadiation_WM2'
]].corr()

# Step 4: Visualize Correlations

# Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Scatter plot for Solar Radiation vs. Solar Energy Production
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['SolarRadiation_WM2'], merged_data['Solar_MW'], alpha=0.5)
plt.title('Solar Radiation vs. Solar Energy Production')
plt.xlabel('Solar Radiation (W/m²)')
plt.ylabel('Solar Energy Production (MW)')
plt.grid(True)
plt.show()

# Scatter plot for Wind Speed vs. Wind Energy Production
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['AvgWindSpeed_kmh'], merged_data['Wind_MW'], alpha=0.5)
plt.title('Wind Speed vs. Wind Energy Production')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Wind Energy Production (MW)')
plt.grid(True)
plt.show()

# Save the merged dataset
merged_data.to_csv('merged_energy_weather_data.csv')

# Provide the path to the saved file
print("Merged dataset saved to: merged_energy_weather_data.csv")



