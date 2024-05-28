"""
This script predicts energy demand and production using weather data.
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
y_import = merged_data['Import_Positive_MW']
y_export = merged_data['Export_Positive_MW']

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
import_model = RandomForestRegressor(random_state=42)
export_model = RandomForestRegressor(random_state=42)

# Align indices after filtering out zero solar production
X = X.reset_index(drop=True)
y_solar = y_solar.reset_index(drop=True)
y_wind = y_wind.reset_index(drop=True)
y_coal = y_coal.reset_index(drop=True)
y_hydrocarbons = y_hydrocarbons.reset_index(drop=True)
y_water = y_water.reset_index(drop=True)
y_nuclear = y_nuclear.reset_index(drop=True)
y_biomass = y_biomass.reset_index(drop=True)
y_import = y_import.reset_index(drop=True)
y_export = y_export.reset_index(drop=True)

# Training and prediction for all energy productions
X_train_solar, X_test_solar, y_train_solar, y_test_solar = train_test_split(X, y_solar, test_size=0.2, random_state=42)
X_train_wind, X_test_wind, y_train_wind, y_test_wind = train_test_split(X, y_wind, test_size=0.2, random_state=42)
X_train_coal, X_test_coal, y_train_coal, y_test_coal = train_test_split(X, y_coal, test_size=0.2, random_state=42)
X_train_hydrocarbons, X_test_hydrocarbons, y_train_hydrocarbons, y_test_hydrocarbons = train_test_split(X, y_hydrocarbons, test_size=0.2, random_state=42)
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(X, y_water, test_size=0.2, random_state=42)
X_train_nuclear, X_test_nuclear, y_train_nuclear, y_test_nuclear = train_test_split(X, y_nuclear, test_size=0.2, random_state=42)
X_train_biomass, X_test_biomass, y_train_biomass, y_test_biomass = train_test_split(X, y_biomass, test_size=0.2, random_state=42)
X_train_import, X_test_import, y_train_import, y_test_import = train_test_split(X, y_import, test_size=0.2, random_state=42)
X_train_export, X_test_export, y_train_export, y_test_export = train_test_split(X, y_export, test_size=0.2, random_state=42)

solar_model.fit(X_train_solar, y_train_solar)
wind_model.fit(X_train_wind, y_train_wind)
coal_model.fit(X_train_coal, y_train_coal)
hydrocarbons_model.fit(X_train_hydrocarbons, y_train_hydrocarbons)
water_model.fit(X_train_water, y_train_water)
nuclear_model.fit(X_train_nuclear, y_train_nuclear)
biomass_model.fit(X_train_biomass, y_train_biomass)
import_model.fit(X_train_import, y_train_import)
export_model.fit(X_train_export, y_train_export)

y_pred_solar = solar_model.predict(X_test_solar)
y_pred_wind = wind_model.predict(X_test_wind)
y_pred_coal = coal_model.predict(X_test_coal)
y_pred_hydrocarbons = hydrocarbons_model.predict(X_test_hydrocarbons)
y_pred_water = water_model.predict(X_test_water)
y_pred_nuclear = nuclear_model.predict(X_test_nuclear)
y_pred_biomass = biomass_model.predict(X_test_biomass)
y_pred_import = import_model.predict(X_test_import)
y_pred_export = export_model.predict(X_test_export)

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
import_rmse = rmse_percentage(y_test_import, y_pred_import)
export_rmse = rmse_percentage(y_test_export, y_pred_export)

# Output RMSE percentages
print(f"Demand Forecast RMSE (%): {demand_rmse}")
print(f"Solar Production Forecast RMSE (%): {solar_rmse}")
print(f"Wind Production Forecast RMSE (%): {wind_rmse}")
print(f"Coal Production Forecast RMSE (%): {coal_rmse}")
print(f"Hydrocarbons Production Forecast RMSE (%): {hydrocarbons_rmse}")
print(f"Water Production Forecast RMSE (%): {water_rmse}")
print(f"Nuclear Production Forecast RMSE (%): {nuclear_rmse}")
print(f"Biomass Production Forecast RMSE (%): {biomass_rmse}")
print(f"Import Forecast RMSE (%): {import_rmse}")
print(f"Export Forecast RMSE (%): {export_rmse}")

# Calculate relative errors for forecasts
demand_relative_error = relative_error(y_test, y_pred_demand)
solar_relative_error = relative_error(y_test_solar, y_pred_solar_corrected)
wind_relative_error = relative_error(y_test_wind, y_pred_wind)
coal_relative_error = relative_error(y_test_coal, y_pred_coal)
hydrocarbons_relative_error = relative_error(y_test_hydrocarbons, y_pred_hydrocarbons)
water_relative_error = relative_error(y_test_water, y_pred_water)
nuclear_relative_error = relative_error(y_test_nuclear, y_pred_nuclear)
biomass_relative_error = relative_error(y_test_biomass, y_pred_biomass)
import_relative_error = relative_error(y_test_import, y_pred_import)
export_relative_error = relative_error(y_test_export, y_pred_export)

# Create dataframes for actual and predicted values
df_demand = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_demand})
df_solar = pd.DataFrame({'Actual': y_test_solar, 'Predicted': y_pred_solar_corrected})
df_wind = pd.DataFrame({'Actual': y_test_wind, 'Predicted': y_pred_wind})
df_coal = pd.DataFrame({'Actual': y_test_coal, 'Predicted': y_pred_coal})
df_hydrocarbons = pd.DataFrame({'Actual': y_test_hydrocarbons, 'Predicted': y_pred_hydrocarbons})
df_water = pd.DataFrame({'Actual': y_test_water, 'Predicted': y_pred_water})
df_nuclear = pd.DataFrame({'Actual': y_test_nuclear, 'Predicted': y_pred_nuclear})
df_biomass = pd.DataFrame({'Actual': y_test_biomass, 'Predicted': y_pred_biomass})
df_import = pd.DataFrame({'Actual': y_test_import, 'Predicted': y_pred_import})
df_export = pd.DataFrame({'Actual': y_test_export, 'Predicted': y_pred_export})

# Create line graphs for actual vs. predicted values
def plot_actual_vs_predicted(df, title):
    plt.figure(figsize=(10, 7))
    plt.plot(df['Actual'].values, label='Actual')
    plt.plot(df['Predicted'].values, label='Predicted')
    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=14) 
    plt.ylabel('Values', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

plot_actual_vs_predicted(df_demand, 'Energy Demand: Actual vs. Predicted')
plot_actual_vs_predicted(df_solar, 'Solar Energy Production: Actual vs. Predicted')
plot_actual_vs_predicted(df_wind, 'Wind Energy Production: Actual vs. Predicted')
plot_actual_vs_predicted(df_coal, 'Coal Energy Production: Actual vs. Predicted')
plot_actual_vs_predicted(df_hydrocarbons, 'Hydrocarbons Energy Production: Actual vs. Predicted')
plot_actual_vs_predicted(df_water, 'Water Energy Production: Actual vs. Predicted')
plot_actual_vs_predicted(df_nuclear, 'Nuclear Energy Production: Actual vs. Predicted')
plot_actual_vs_predicted(df_biomass, 'Biomass Energy Production: Actual vs. Predicted')
plot_actual_vs_predicted(df_import, 'Import: Actual vs. Predicted')
plot_actual_vs_predicted(df_export, 'Export: Actual vs. Predicted')

# Scatter Plots for Actual vs. Predicted Values

def plot_scatter(df, title, x_label, y_label, color):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=df['Actual'], y=df['Predicted'], color=color, label='Predicted')
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

plot_scatter(df_demand, 'Energy Demand: Actual vs. Predicted', 'Actual Demand (MW)', 'Predicted Demand (MW)', 'blue')
plot_scatter(df_solar, 'Solar Energy Production: Actual vs. Predicted', 'Actual Solar Production (MW)', 'Predicted Solar Production (MW)', 'green')
plot_scatter(df_wind, 'Wind Energy Production: Actual vs. Predicted', 'Actual Wind Production (MW)', 'Predicted Wind Production (MW)', 'purple')
plot_scatter(df_coal, 'Coal Energy Production: Actual vs. Predicted', 'Actual Coal Production (MW)', 'Predicted Coal Production (MW)', 'brown')
plot_scatter(df_hydrocarbons, 'Hydrocarbons Energy Production: Actual vs. Predicted', 'Actual Hydrocarbons Production (MW)', 'Predicted Hydrocarbons Production (MW)', 'orange')
plot_scatter(df_water, 'Water Energy Production: Actual vs. Predicted', 'Actual Water Production (MW)', 'Predicted Water Production (MW)', 'blue')
plot_scatter(df_nuclear, 'Nuclear Energy Production: Actual vs. Predicted', 'Actual Nuclear Production (MW)', 'Predicted Nuclear Production (MW)', 'red')
plot_scatter(df_biomass, 'Biomass Energy Production: Actual vs. Predicted', 'Actual Biomass Production (MW)', 'Predicted Biomass Production (MW)', 'green')
plot_scatter(df_import, 'Import: Actual vs. Predicted', 'Actual Import (MW)', 'Predicted Import (MW)', 'cyan')
plot_scatter(df_export, 'Export: Actual vs. Predicted', 'Actual Export (MW)', 'Predicted Export (MW)', 'magenta')

# Create dataframes for relative errors
df_relative_error_demand = pd.DataFrame({'Relative_Error': demand_relative_error})
df_relative_error_solar = pd.DataFrame({'Relative_Error': solar_relative_error})
df_relative_error_wind = pd.DataFrame({'Relative_Error': wind_relative_error})
df_relative_error_coal = pd.DataFrame({'Relative_Error': coal_relative_error})
df_relative_error_hydrocarbons = pd.DataFrame({'Relative_Error': hydrocarbons_relative_error})
df_relative_error_water = pd.DataFrame({'Relative_Error': water_relative_error})
df_relative_error_nuclear = pd.DataFrame({'Relative_Error': nuclear_relative_error})
df_relative_error_biomass = pd.DataFrame({'Relative_Error': biomass_relative_error})
df_relative_error_import = pd.DataFrame({'Relative_Error': import_relative_error})
df_relative_error_export = pd.DataFrame({'Relative_Error': export_relative_error})

# Create line graphs for relative errors
def plot_relative_error(df, title):
    plt.figure(figsize=(10, 7))
    plt.plot(df['Relative_Error'].values)
    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Relative Error (%)', fontsize=14)
    plt.show()

plot_relative_error(df_relative_error_demand, 'Relative Error: Energy Demand')
plot_relative_error(df_relative_error_solar, 'Relative Error: Solar Energy Production')
plot_relative_error(df_relative_error_wind, 'Relative Error: Wind Energy Production')
plot_relative_error(df_relative_error_coal, 'Relative Error: Coal Energy Production')
plot_relative_error(df_relative_error_hydrocarbons, 'Relative Error: Hydrocarbons Energy Production')
plot_relative_error(df_relative_error_water, 'Relative Error: Water Energy Production')
plot_relative_error(df_relative_error_nuclear, 'Relative Error: Nuclear Energy Production')
plot_relative_error(df_relative_error_biomass, 'Relative Error: Biomass Energy Production')
plot_relative_error(df_relative_error_import, 'Relative Error: Import')
plot_relative_error(df_relative_error_export, 'Relative Error: Export')

# Create dataframes for actual and predicted values
df_predictions = pd.DataFrame({
    'Date': merged_data.index[X_test.index],
    'Predicted_Demand_MW': y_pred_demand,
    'Predicted_Solar_MW': y_pred_solar_corrected,
    'Predicted_Wind_MW': y_pred_wind,
    'Predicted_Coal_MW': y_pred_coal,
    'Predicted_Hydrocarbons_MW': y_pred_hydrocarbons,
    'Predicted_Water_MW': y_pred_water,
    'Predicted_Nuclear_MW': y_pred_nuclear,
    'Predicted_Biomass_MW': y_pred_biomass,
    'Predicted_Import_MW': y_pred_import,
    'Predicted_Export_MW': y_pred_export
})

# Save predictions to CSV
df_predictions.to_csv('energy_predictions.csv', index=False)

print("Predictions saved to energy_predictions.csv")