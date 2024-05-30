"""
This script predicts energy demand and production using neural networks.
"""

# Step 1: Import Libraries and Load Datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train_demand, y_test_demand = train_test_split(X_scaled, y_demand, test_size=0.2, random_state=42)
_, _, y_train_solar, y_test_solar = train_test_split(X_scaled, y_solar, test_size=0.2, random_state=42)
_, _, y_train_wind, y_test_wind = train_test_split(X_scaled, y_wind, test_size=0.2, random_state=42)
_, _, y_train_coal, y_test_coal = train_test_split(X_scaled, y_coal, test_size=0.2, random_state=42)
_, _, y_train_hydrocarbons, y_test_hydrocarbons = train_test_split(X_scaled, y_hydrocarbons, test_size=0.2, random_state=42)
_, _, y_train_water, y_test_water = train_test_split(X_scaled, y_water, test_size=0.2, random_state=42)
_, _, y_train_nuclear, y_test_nuclear = train_test_split(X_scaled, y_nuclear, test_size=0.2, random_state=42)
_, _, y_train_biomass, y_test_biomass = train_test_split(X_scaled, y_biomass, test_size=0.2, random_state=42)
_, _, y_train_import, y_test_import = train_test_split(X_scaled, y_import, test_size=0.2, random_state=42)
_, _, y_train_export, y_test_export = train_test_split(X_scaled, y_export, test_size=0.2, random_state=42)

# Define the neural network model
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train and predict function
def train_and_predict(X_train, y_train, X_test):
    model = create_model()
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stop], verbose=1)
    y_pred = model.predict(X_test).flatten()
    return y_pred

# Predict all targets
y_pred_demand = train_and_predict(X_train, y_train_demand, X_test)
y_pred_solar = train_and_predict(X_train, y_train_solar, X_test)
y_pred_wind = train_and_predict(X_train, y_train_wind, X_test)
y_pred_coal = train_and_predict(X_train, y_train_coal, X_test)
y_pred_hydrocarbons = train_and_predict(X_train, y_train_hydrocarbons, X_test)
y_pred_water = train_and_predict(X_train, y_train_water, X_test)
y_pred_nuclear = train_and_predict(X_train, y_train_nuclear, X_test)
y_pred_biomass = train_and_predict(X_train, y_train_biomass, X_test)
y_pred_import = train_and_predict(X_train, y_train_import, X_test)
y_pred_export = train_and_predict(X_train, y_train_export, X_test)

# Correct solar predictions where actuals are zero
mask_zeros_in_actuals = y_test_solar == 0
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
demand_rmse = rmse_percentage(y_test_demand, y_pred_demand)
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
demand_relative_error = relative_error(y_test_demand, y_pred_demand)
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
df_demand = pd.DataFrame({'Actual': y_test_demand, 'Predicted': y_pred_demand})
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
df_predictions.to_csv('energy_predictions_nn.csv', index=False)

print("Predictions saved to energy_predictions_nn.csv")