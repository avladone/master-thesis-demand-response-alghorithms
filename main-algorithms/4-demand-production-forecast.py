"""
This script forecasts energy demand and production using weather data.
"""

# Step 1: Import Libraries and Load Datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load merged dataset
merged_data = pd.read_csv('merged_energy_weather_data.csv')
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.set_index('Date', inplace=True)

# Forecasting setup excluding zero solar production
X = merged_data[['AvgTemp_Celsius', 'AvgWindSpeed_kmh', 'SolarRadiation_WM2']]
y_demand = merged_data['Demand_MW']
y_solar = merged_data['Solar_MW']
y_wind = merged_data['Wind_MW']
y_coal = merged_data['Coal_MW']
y_hydrocarbons = merged_data['Hydrocarbons_MW']
y_water = merged_data['Water_MW']
y_nuclear = merged_data['Nuclear_MW']
y_biomass = merged_data['Biomass_MW']

# Demand forecasting model
X_train, X_test, y_train, y_test = train_test_split(X, y_demand, test_size=0.2, random_state=42)
demand_model = RandomForestRegressor(random_state=42)
demand_model.fit(X_train, y_train)
y_pred_demand = demand_model.predict(X_test)

# Setup for all energy production models
solar_model = RandomForestRegressor(random_state=42)
wind_model = RandomForestRegressor(random_state=42)
coal_model = RandomForestRegressor(random_state=42)
hydrocarbons_model = RandomForestRegressor(random_state=42)
water_model = RandomForestRegressor(random_state=42)
nuclear_model = RandomForestRegressor(random_state=42)
biomass_model = RandomForestRegressor(random_state=42)

# Align indices after filtering out zero solar production
X = X.reset_index(drop=True)
y_solar = y_solar.reset_index(drop=True)
y_wind = y_wind.reset_index(drop=True)
y_coal = y_coal.reset_index(drop=True)
y_hydrocarbons = y_hydrocarbons.reset_index(drop=True)
y_water = y_water.reset_index(drop=True)
y_nuclear = y_nuclear.reset_index(drop=True)
y_biomass = y_biomass.reset_index(drop=True)

# Training and prediction for all energy productions
X_train_solar, X_test_solar, y_train_solar, y_test_solar = train_test_split(X, y_solar, test_size=0.2, random_state=42)
X_train_wind, X_test_wind, y_train_wind, y_test_wind = train_test_split(X, y_wind, test_size=0.2, random_state=42)
X_train_coal, X_test_coal, y_train_coal, y_test_coal = train_test_split(X, y_coal, test_size=0.2, random_state=42)
X_train_hydrocarbons, X_test_hydrocarbons, y_train_hydrocarbons, y_test_hydrocarbons = train_test_split(X, y_hydrocarbons, test_size=0.2, random_state=42)
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(X, y_water, test_size=0.2, random_state=42)
X_train_nuclear, X_test_nuclear, y_train_nuclear, y_test_nuclear = train_test_split(X, y_nuclear, test_size=0.2, random_state=42)
X_train_biomass, X_test_biomass, y_train_biomass, y_test_biomass = train_test_split(X, y_biomass, test_size=0.2, random_state=42)

solar_model.fit(X_train_solar, y_train_solar)
wind_model.fit(X_train_wind, y_train_wind)
coal_model.fit(X_train_coal, y_train_coal)
hydrocarbons_model.fit(X_train_hydrocarbons, y_train_hydrocarbons)
water_model.fit(X_train_water, y_train_water)
nuclear_model.fit(X_train_nuclear, y_train_nuclear)
biomass_model.fit(X_train_biomass, y_train_biomass)

y_pred_solar = solar_model.predict(X_test_solar)
y_pred_wind = wind_model.predict(X_test_wind)
y_pred_coal = coal_model.predict(X_test_coal)
y_pred_hydrocarbons = hydrocarbons_model.predict(X_test_hydrocarbons)
y_pred_water = water_model.predict(X_test_water)
y_pred_nuclear = nuclear_model.predict(X_test_nuclear)
y_pred_biomass = biomass_model.predict(X_test_biomass)

# Correct solar predictions where actuals are zero
actual_zeros = merged_data[merged_data['Solar_MW'] == 0].index
mask_zeros_in_actuals = X_test_solar.index.isin(actual_zeros)
y_pred_solar_corrected = np.where(mask_zeros_in_actuals, 0, y_pred_solar)

# Function to calculate RMSE percentage
def rmse_percentage(true_values, predicted_values):
    rmse = sqrt(mean_squared_error(true_values, predicted_values))
    average = np.mean(true_values)
    return (rmse / average) * 100

# Function to calculate relative error
def relative_error(true_values, predicted_values):
    return ((true_values - predicted_values) / true_values) * 100

# Calculate RMSE for forecasts
demand_rmse = rmse_percentage(y_test, y_pred_demand)
solar_rmse = rmse_percentage(y_test_solar, y_pred_solar_corrected)
wind_rmse = rmse_percentage(y_test_wind, y_pred_wind)
coal_rmse = rmse_percentage(y_test_coal, y_pred_coal)
hydrocarbons_rmse = rmse_percentage(y_test_hydrocarbons, y_pred_hydrocarbons)
water_rmse = rmse_percentage(y_test_water, y_pred_water)
nuclear_rmse = rmse_percentage(y_test_nuclear, y_pred_nuclear)
biomass_rmse = rmse_percentage(y_test_biomass, y_pred_biomass)

# Output RMSE percentages
print(f"Demand Forecast RMSE (%): {demand_rmse}")
print(f"Solar Production Forecast RMSE (%): {solar_rmse}")
print(f"Wind Production Forecast RMSE (%): {wind_rmse}")
print(f"Coal Production Forecast RMSE (%): {coal_rmse}")
print(f"Hydrocarbons Production Forecast RMSE (%): {hydrocarbons_rmse}")
print(f"Water Production Forecast RMSE (%): {water_rmse}")
print(f"Nuclear Production Forecast RMSE (%): {nuclear_rmse}")
print(f"Biomass Production Forecast RMSE (%): {biomass_rmse}")

# Calculate relative errors for forecasts
demand_relative_error = relative_error(y_test, y_pred_demand)
solar_relative_error = relative_error(y_test_solar, y_pred_solar_corrected)
wind_relative_error = relative_error(y_test_wind, y_pred_wind)
coal_relative_error = relative_error(y_test_coal, y_pred_coal)
hydrocarbons_relative_error = relative_error(y_test_hydrocarbons, y_pred_hydrocarbons)
water_relative_error = relative_error(y_test_water, y_pred_water)
nuclear_relative_error = relative_error(y_test_nuclear, y_pred_nuclear)
biomass_relative_error = relative_error(y_test_biomass, y_pred_biomass)

# Step 5: Visualization of Forecasts vs. Actuals

# Energy Demand Visualization
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test, y=y_pred_demand, color='blue', label='Predicted')
plt.title('Energy Demand: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Demand (MW)', fontsize=14)
plt.ylabel('Predicted Demand (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(y_test.values, label='Actual Demand')
plt.plot(y_pred_demand, label='Predicted Demand')
plt.title('Energy Demand: Actual vs. Predicted', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Demand (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Solar Energy Production Visualization
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test_solar, y=y_pred_solar_corrected, color='green', label='Predicted')
plt.title('Solar Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Solar Production (MW)', fontsize=14)
plt.ylabel('Predicted Solar Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(y_test_solar.values, label='Actual Solar Production')
plt.plot(y_pred_solar_corrected, label='Predicted Solar Production')
plt.title('Solar Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Solar Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Wind Energy Production Visualization
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test_wind, y=y_pred_wind, color='purple', label='Predicted')
plt.title('Wind Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Wind Production (MW)', fontsize=14)
plt.ylabel('Predicted Wind Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(y_test_wind.values, label='Actual Wind Production')
plt.plot(y_pred_wind, label='Predicted Wind Production')
plt.title('Wind Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Wind Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Coal Energy Production Visualization
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test_coal, y=y_pred_coal, color='brown', label='Predicted')
plt.title('Coal Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Coal Production (MW)', fontsize=14)
plt.ylabel('Predicted Coal Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(y_test_coal.values, label='Actual Coal Production')
plt.plot(y_pred_coal, label='Predicted Coal Production')
plt.title('Coal Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Coal Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Hydrocarbons Energy Production Visualization
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test_hydrocarbons, y=y_pred_hydrocarbons, color='orange', label='Predicted')
plt.title('Hydrocarbons Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Hydrocarbons Production (MW)', fontsize=14)
plt.ylabel('Predicted Hydrocarbons Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(y_test_hydrocarbons.values, label='Actual Hydrocarbons Production')
plt.plot(y_pred_hydrocarbons, label='Predicted Hydrocarbons Production')
plt.title('Hydrocarbons Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Hydrocarbons Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Water Energy Production Visualization
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test_water, y=y_pred_water, color='blue', label='Predicted')
plt.title('Water Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Water Production (MW)', fontsize=14)
plt.ylabel('Predicted Water Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(y_test_water.values, label='Actual Water Production')
plt.plot(y_pred_water, label='Predicted Water Production')
plt.title('Water Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Water Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Nuclear Energy Production Visualization
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test_nuclear, y=y_pred_nuclear, color='red', label='Predicted')
plt.title('Nuclear Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Nuclear Production (MW)', fontsize=14)
plt.ylabel('Predicted Nuclear Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(y_test_nuclear.values, label='Actual Nuclear Production')
plt.plot(y_pred_nuclear, label='Predicted Nuclear Production')
plt.title('Nuclear Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Nuclear Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Biomass Energy Production Visualization
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test_biomass, y=y_pred_biomass, color='green', label='Predicted')
plt.title('Biomass Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Biomass Production (MW)', fontsize=14)
plt.ylabel('Predicted Biomass Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(y_test_biomass.values, label='Actual Biomass Production')
plt.plot(y_pred_biomass, label='Predicted Biomass Production')
plt.title('Biomass Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Biomass Production (MW)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Save the merged dataset with predictions
merged_data['Predicted_Demand_MW'] = np.nan
merged_data.loc[X_test.index, 'Predicted_Demand_MW'] = y_pred_demand

merged_data['Predicted_Solar_MW'] = np.nan
merged_data.loc[X_test_solar.index, 'Predicted_Solar_MW'] = y_pred_solar_corrected

merged_data['Predicted_Wind_MW'] = np.nan
merged_data.loc[X_test_wind.index, 'Predicted_Wind_MW'] = y_pred_wind

merged_data['Predicted_Coal_MW'] = np.nan
merged_data.loc[X_test_coal.index, 'Predicted_Coal_MW'] = y_pred_coal

merged_data['Predicted_Hydrocarbons_MW'] = np.nan
merged_data.loc[X_test_hydrocarbons.index, 'Predicted_Hydrocarbons_MW'] = y_pred_hydrocarbons

merged_data['Predicted_Water_MW'] = np.nan
merged_data.loc[X_test_water.index, 'Predicted_Water_MW'] = y_pred_water

merged_data['Predicted_Nuclear_MW'] = np.nan
merged_data.loc[X_test_nuclear.index, 'Predicted_Nuclear_MW'] = y_pred_nuclear

merged_data['Predicted_Biomass_MW'] = np.nan
merged_data.loc[X_test_biomass.index, 'Predicted_Biomass_MW'] = y_pred_biomass

merged_data.to_csv('merged_energy_weather_predictions.csv')

# Provide the path to the saved file
print("Merged dataset with predictions saved to: merged_energy_weather_predictions.csv")