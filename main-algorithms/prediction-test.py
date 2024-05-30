"""
This script predicts energy demand and production using weather data and additional features.
"""

# Step 1: Import Libraries and Load Datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load merged dataset
merged_data = pd.read_csv('merged_energy_weather_data.csv')
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.set_index('Date', inplace=True)

# Create calendar features
merged_data['DayOfWeek'] = merged_data.index.dayofweek
merged_data['Month'] = merged_data.index.month
merged_data['DayOfYear'] = merged_data.index.dayofyear

# Add a hypothetical economic indicator
np.random.seed(42)
merged_data['EconomicIndicator'] = np.random.normal(loc=100, scale=10, size=len(merged_data))

# Create lag features and rolling statistics
def create_lag_features(data, lags, columns):
    for col in columns:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    return data

def create_rolling_features(data, windows, columns):
    for col in columns:
        for window in windows:
            data[f'{col}_roll_mean_{window}'] = data[col].rolling(window=window).mean()
            data[f'{col}_roll_std_{window}'] = data[col].rolling(window=window).std()
    return data

# Specify columns for lag and rolling features
lag_columns = ['Demand_MW', 'Solar_MW', 'Wind_MW', 'Coal_MW', 'Hydrocarbons_MW', 'Water_MW', 'Nuclear_MW', 'Biomass_MW']
lags = [1, 2, 3, 7, 14, 30]
windows = [3, 7, 14, 30]

# Create lag and rolling features
merged_data = create_lag_features(merged_data, lags, lag_columns)
merged_data = create_rolling_features(merged_data, windows, lag_columns)

# Drop rows with NaN values created by lag and rolling features
merged_data = merged_data.dropna()

# Forecasting setup
X = merged_data[['AvgTemp_Celsius', 'AvgWindSpeed_kmh', 'SolarRadiation_WM2', 'DayOfWeek', 'Month', 'DayOfYear', 'EconomicIndicator'] +
                [f'{col}_lag_{lag}' for col in lag_columns for lag in lags] +
                [f'{col}_roll_mean_{window}' for col in lag_columns for window in windows] +
                [f'{col}_roll_std_{window}' for col in lag_columns for window in windows]]
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

# Train-test split
def train_test_split_and_train_model(X, y, model_type='rf'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(random_state=42)
    elif model_type == 'nn':
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stop], verbose=1)
        y_pred = model.predict(X_test).flatten()
        return y_test, y_pred
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred

# Choose model type: 'rf' for Random Forest, 'gb' for Gradient Boosting, 'nn' for Neural Network
model_type = 'rf'  # Change this to 'gb' or 'nn' for different models

# Training and prediction for all energy productions
y_test_demand, y_pred_demand = train_test_split_and_train_model(X, y_demand, model_type)
y_test_solar, y_pred_solar = train_test_split_and_train_model(X, y_solar, model_type)
y_test_wind, y_pred_wind = train_test_split_and_train_model(X, y_wind, model_type)
y_test_coal, y_pred_coal = train_test_split_and_train_model(X, y_coal, model_type)
y_test_hydrocarbons, y_pred_hydrocarbons = train_test_split_and_train_model(X, y_hydrocarbons, model_type)
y_test_water, y_pred_water = train_test_split_and_train_model(X, y_water, model_type)
y_test_nuclear, y_pred_nuclear = train_test_split_and_train_model(X, y_nuclear, model_type)
y_test_biomass, y_pred_biomass = train_test_split_and_train_model(X, y_biomass, model_type)
y_test_import, y_pred_import = train_test_split_and_train_model(X, y_import, model_type)
y_test_export, y_pred_export = train_test_split_and_train_model(X, y_export, model_type)

# Function to calculate RMSE percentage
def rmse_percentage(true_values, predicted_values):
    rmse = sqrt(mean_squared_error(true_values, predicted_values))
    average = np.mean(true_values)
    return (rmse / average) * 100

# Calculate RMSE for forecasts
demand_rmse = rmse_percentage(y_test_demand, y_pred_demand)
solar_rmse = rmse_percentage(y_test_solar, y_pred_solar)
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

# Plotting the relative errors
def plot_relative_error(y_test, y_pred, title):
    relative_error = ((y_test - y_pred) / y_test) * 100
    plt.figure(figsize=(10, 6))
    plt.plot(relative_error, label='Relative Error')
    plt.title(f'Relative Error: {title}')
    plt.xlabel('Time')
    plt.ylabel('Relative Error (%)')
    plt.legend()
    plt.show()

plot_relative_error(y_test_demand, y_pred_demand, 'Energy Demand')
plot_relative_error(y_test_solar, y_pred_solar, 'Solar Energy Production')
plot_relative_error(y_test_wind, y_pred_wind, 'Wind Energy Production')
plot_relative_error(y_test_coal, y_pred_coal, 'Coal Energy Production')
plot_relative_error(y_test_hydrocarbons, y_pred_hydrocarbons, 'Hydrocarbons Energy Production')
plot_relative_error(y_test_water, y_pred_water, 'Water Energy Production')
plot_relative_error(y_test_nuclear, y_pred_nuclear, 'Nuclear Energy Production')
plot_relative_error(y_test_biomass, y_pred_biomass, 'Biomass Energy Production')
plot_relative_error(y_test_import, y_pred_import, 'Import')
plot_relative_error(y_test_export, y_pred_export, 'Export')

# Create dataframes for actual and predicted values
df_demand = pd.DataFrame({'Actual': y_test_demand, 'Predicted': y_pred_demand})
df_solar = pd.DataFrame({'Actual': y_test_solar, 'Predicted': y_pred_solar})
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

# Function to calculate relative error
def relative_error(true_values, predicted_values):
    return ((true_values - predicted_values) / true_values) * 100

# Create dataframes for relative errors
df_relative_error_demand = pd.DataFrame({'Relative_Error': relative_error(y_test_demand, y_pred_demand)})
df_relative_error_solar = pd.DataFrame({'Relative_Error': relative_error(y_test_solar, y_pred_solar)})
df_relative_error_wind = pd.DataFrame({'Relative_Error': relative_error(y_test_wind, y_pred_wind)})
df_relative_error_coal = pd.DataFrame({'Relative_Error': relative_error(y_test_coal, y_pred_coal)})
df_relative_error_hydrocarbons = pd.DataFrame({'Relative_Error': relative_error(y_test_hydrocarbons, y_pred_hydrocarbons)})
df_relative_error_water = pd.DataFrame({'Relative_Error': relative_error(y_test_water, y_pred_water)})
df_relative_error_nuclear = pd.DataFrame({'Relative_Error': relative_error(y_test_nuclear, y_pred_nuclear)})
df_relative_error_biomass = pd.DataFrame({'Relative_Error': relative_error(y_test_biomass, y_pred_biomass)})
df_relative_error_import = pd.DataFrame({'Relative_Error': relative_error(y_test_import, y_pred_import)})
df_relative_error_export = pd.DataFrame({'Relative_Error': relative_error(y_test_export, y_pred_export)})

# Create line graphs for relative errors
def plot_relative_error_line(df, title):
    plt.figure(figsize=(10, 7))
    plt.plot(df['Relative_Error'].values)
    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Relative Error (%)', fontsize=14)
    plt.show()

plot_relative_error_line(df_relative_error_demand, 'Relative Error: Energy Demand')
plot_relative_error_line(df_relative_error_solar, 'Relative Error: Solar Energy Production')
plot_relative_error_line(df_relative_error_wind, 'Relative Error: Wind Energy Production')
plot_relative_error_line(df_relative_error_coal, 'Relative Error: Coal Energy Production')
plot_relative_error_line(df_relative_error_hydrocarbons, 'Relative Error: Hydrocarbons Energy Production')
plot_relative_error_line(df_relative_error_water, 'Relative Error: Water Energy Production')
plot_relative_error_line(df_relative_error_nuclear, 'Relative Error: Nuclear Energy Production')
plot_relative_error_line(df_relative_error_biomass, 'Relative Error: Biomass Energy Production')
plot_relative_error_line(df_relative_error_import, 'Relative Error: Import')
plot_relative_error_line(df_relative_error_export, 'Relative Error: Export')

# Save predictions to CSV
df_predictions = pd.DataFrame({
    'Predicted_Demand_MW': y_pred_demand,
    'Predicted_Solar_MW': y_pred_solar,
    'Predicted_Wind_MW': y_pred_wind,
    'Predicted_Coal_MW': y_pred_coal,
    'Predicted_Hydrocarbons_MW': y_pred_hydrocarbons,
    'Predicted_Water_MW': y_pred_water,
    'Predicted_Nuclear_MW': y_pred_nuclear,
    'Predicted_Biomass_MW': y_pred_biomass,
    'Predicted_Import_MW': y_pred_import,
    'Predicted_Export_MW': y_pred_export
})

df_predictions.to_csv('energy_predictions_enhanced.csv', index=False)

print("Predictions saved to energy_predictions_enhanced.csv")