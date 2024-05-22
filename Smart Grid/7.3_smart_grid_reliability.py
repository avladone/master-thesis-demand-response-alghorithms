# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:14:46 2024

This case study focuses on using Python
for historical data analysis to identify energy patterns, forecasting demand 
and renewable production, and devising optimization strategies for reliable 
grid management. The goal is to improve grid stability while maximizing 
renewable energy use, assessing both environmental and economic impacts.
"""

# Step 1: Import Libraries and Load Datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pulp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load datasets
energy_data = pd.read_csv('EnergyData_backup.csv')
weather_data = pd.read_csv('WeatherData.csv')

# Step 2: Preliminary Data Analysis
print(energy_data.head())
print(weather_data.head())

# Analyzing patterns in energy demand and production by month
energy_data['Date'] = pd.to_datetime(energy_data['Date'])
energy_data.set_index('Date', inplace=True)
energy_data.groupby(energy_data.index.month).mean().plot(
    kind='bar', y=['Demand_MW', "Solar_MW", "Wind_MW"]
    )
plt.title("Average Energy Demand and Production by Month")
plt.show()

# Step 3: Merge Datasets and Correlation Analysis
# Reset index for merging
energy_data.reset_index(inplace=True)

# Ensure 'Date' columns are in datetime format
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Merge datasets on 'Date'
merged_data = pd.merge(energy_data, weather_data, on='Date')

# Filter non-zero solar production 
merged_data = merged_data[merged_data['Solar_MW'] > 0]

# Correlation of weather conditions and energy production
correlation = merged_data[[
    'AvgTemp_Celsius', 'AvgWindSpeed_kmh',
    'Solar_MW', 'Wind_MW'
    ]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)

# Step 4: Forecasting Energy Demand and Production
# Forecasting setup excluding zero solar production
X = merged_data[['AvgTemp_Celsius', 'AvgWindSpeed_kmh']]
y_demand = merged_data['Demand_MW']
y_solar = merged_data['Solar_MW']
y_wind = merged_data['Wind_MW']

# Demand forecasting model
X_train, X_test, y_train, y_test = train_test_split(
    X, y_demand, test_size=0.2, random_state=42
    )
demand_model = RandomForestRegressor(random_state=42)
demand_model.fit(X_train, y_train)
y_pred_demand = demand_model.predict(X_test)

# Setup for solar and wind models
solar_model = RandomForestRegressor(random_state=42)
wind_model = RandomForestRegressor(random_state=42)

# Align indices after filtering out zero solar production
X = X.reset_index(drop=True)
y_solar = y_solar.reset_index(drop=True)
y_wind = y_wind.reset_index(drop=True)

# Training and prediction for solar and wind energy
X_train_solar, X_test_solar, \
y_train_solar, y_test_solar = \
    train_test_split(X, y_solar, test_size=0.2, random_state=42)
    
X_train_wind, X_test_wind, \
y_train_wind, y_test_wind = \
    train_test_split(X, y_wind, test_size = 0.2, random_state=42)
    
solar_model.fit(X_train_solar, y_train_solar)
wind_model.fit(X_train_wind, y_train_wind)
y_pred_solar = solar_model.predict(X_test_solar)
y_pred_wind = wind_model.predict(X_test_wind)

# Correct solar predictions where actuals are zero
actual_zeros = energy_data[energy_data['Solar_MW'] == 0].index
mask_zeros_in_actuals = X_test_solar.index.isin(actual_zeros)
y_pred_solar_corrected = np.where(
    mask_zeros_in_actuals, 0, y_pred_solar
    )

# Function to calculare RMSE percentage
def rmse_percentage(true_values, predicted_values):
    rmse = sqrt(mean_squared_error(true_values, predicted_values))
    average = np.mean(true_values)
    return (rmse/average) * 100

# Calculate RMSE for forecasts including zero solar production
demand_rmse = rmse_percentage(y_test, y_pred_demand)
solar_rmse = rmse_percentage(y_test_solar, y_pred_solar)
wind_rmse = rmse_percentage(y_test_wind, y_pred_wind)

# Output RMSE percentages
print(f"Demand Forecast RMSE (%): {demand_rmse}")
print(f"Solar Production Forecast RMSE (%): {solar_rmse}")
print(f"Wind Production Forecast RMSE (%): {wind_rmse}")

# Step 5: Visualization of Forecasts vs. Actuals

# Visualization of Forecasts vs. Actuals for Energy Metrics

# Energy Demand Visualization
plt.figure(figsize=(10,7))
sns.scatterplot(
    x=y_test, y=y_pred_demand, color='blue', label='Predicted'
    )
plt.title('Energy Demand: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Demand', fontsize=14)
plt.ylabel('Predicted Demand', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Solar Energy Production Visualization
plt.figure(figsize=(10,7))
sns.scatterplot(
    x=y_test_solar, y=y_pred_solar, color='green', label='Predicted'
    )
plt.title('Solar Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Energy Production', fontsize=14)
plt.ylabel('Predicted Energy Production', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Wind Energy Production Visualization
plt.figure(figsize=(10,7))
sns.scatterplot(
    x=y_test_wind, y=y_pred_wind, color='purple', label='Predicted'
    )
plt.title('Wind Energy Production: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual Wind Production', fontsize=14)
plt.ylabel('Predicted Wind Production', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Step 6: Daily Energy Forecasting with SARIMA
energy_data_daily = energy_data.set_index('Date').resample('D').mean()
sarima_model = SARIMAX(
    energy_data_daily['Demand_MW'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)
    )
sarima_results = sarima_model.fit(disp=False)
sarima_forecast = sarima_results.forecast(steps=7)
print("SARIMA Forecast for the next 7 days:")
print(sarima_forecast)

# Step 7: Optimization with PuLP
avg_solar_production_mwh_corrected = np.mean(y_pred_solar_corrected)
avg_wind_production_mwh = np.mean(y_pred_wind)
avg_daily_demand_mw = np.mean(y_pred_demand)

# Constants for Impact Assessment
grid_emission_factor = 0.5
solar_cost_per_kwh = 0.05
wind_cost_per_kwh = 0.07
natural_gas_cost_per_kwh = 0.15

# Define optimization problem
problem = pulp.LpProblem("Energy_Optimization", pulp.LpMinimize)

# Update the supply variable to use the corrected solar production
grid_supply = pulp.LpVariable("Grid_Supply", lowBound=0)

solar_supply = pulp.LpVariable(
    "Solar_Supply",
    lowBound=0,
    upBound=avg_solar_production_mwh_corrected
    )
wind_supply = pulp.LpVariable(
    "Wind_Supply",
    lowBound=0,
    upBound=avg_wind_production_mwh
    )

# Objective Function
problem += (
    solar_cost_per_kwh * solar_supply + 
    wind_cost_per_kwh * wind_supply + 
    natural_gas_cost_per_kwh * grid_supply,
    "Total Cost"
    )

problem += (
    grid_supply + solar_supply + wind_supply >= avg_daily_demand_mw,
    "Demand_Meeting"
    )
problem.solve()

# Check solution status
if pulp.LpStatus[problem.status] == 'Optimal':
    print(f"Optimal grid supply: {grid_supply.varValue} MW")
    print(f"Optimal solar supply: {solar_supply.varValue} MW")
    print(f"Optimal wind supply: {wind_supply.varValue} MW")
else:
    print("Optimization did not find a feasible solution.")
    
# Step 8: Impact Assessment and Visualization
emissions_saved_kg = (
    solar_supply.varValue + wind_supply.varValue
    ) * grid_emission_factor
cost_with_gas = avg_daily_demand_mw * natural_gas_cost_per_kwh
cost_with_optimization = pulp.value(problem.objective)
savings = cost_with_gas - cost_with_optimization

print(f"Emissions Saved: {emissions_saved_kg} kg CO2")
print(f"Economic Savings: ${savings}")

# Visualization
plt.figure(figsize=(10,6))
plt.bar(
        ['Grid', 'Solar', 'Wind'],
        [grid_supply.varValue, solar_supply.varValue, wind_supply.varValue],
        color=['blue', 'orange', 'green']
        )
plt.title('Optimized Energy Supply Distribution')
plt.xlabel('Energy Source')
plt.ylabel('Energy Supplied (MW)')
plt.show()

plt.figure(figsize=(10,6))
plt.bar(
        ['CO2 Emissions Saved (kg)', 'Economic Savings ($)'],
        [emissions_saved_kg, savings],
        color=['red', 'green']
        )
plt.title('Environmental and Economic Impact of Optimization')
plt.show()
