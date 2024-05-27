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

# Function to calculate RMSE percentage
def rmse_percentage(true_values, predicted_values):
    rmse = sqrt(mean_squared_error(true_values, predicted_values))
    average = np.mean(true_values)
    return (rmse/average) * 100

# Load energy dataset
energy_data = pd.read_csv('EnergyData_2023_new.csv')
energy_data['Date'] = pd.to_datetime(energy_data['Date'])
energy_data.set_index('Date', inplace=True)

# Load weather dataset
weather_data = pd.read_csv('WeatherData_new.csv')
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
weather_data.set_index('Date', inplace=True)

# Step 1: Ensure Continuous DateTime Index
# Remove duplicate timestamps
energy_data = energy_data[~energy_data.index.duplicated(keep='first')]

# Create a complete DateTime index for the entire year
full_index = pd.date_range(start='2023-01-01', end='2023-12-31 23:59:59', freq='H')

# Reindex energy_data to have a continuous DateTime index
energy_data = energy_data.reindex(full_index)

# Step 2: Handle Missing Values
# Fill missing values using forward fill, then backward fill for any remaining NaNs
energy_data.ffill(inplace=True)
energy_data.bfill(inplace=True)

# Resample energy data to hourly mean values
energy_hourly = energy_data.resample('H').mean()

# Verify the number of rows in energy_hourly
print(f"Number of rows in energy_hourly: {len(energy_hourly)}")

# Step 2: Preliminary Data Analysis

# Print the first few rows of each dataset
print(energy_data.head())
print(weather_data.head())

# Plotting Energy Demand and Production by Hour
plt.figure(figsize=(15,10))
energy_hourly.plot(
    kind='line', y=["Demand_MW", "Total_Production_MW", "Coal_MW", 
                    "Hydrocarbons_MW", "Water_MW", "Nuclear_MW", 
                    "Wind_MW", "Solar_MW", "Biomass_MW", "Import_MW"]
)
plt.xlabel("Hour")
plt.ylabel("MW")
plt.title("Energy Demand and Production by Hour")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Resample energy data to daily and monthly mean values
energy_daily = energy_hourly.resample('D').mean()
energy_monthly = energy_hourly.resample('M').mean()

# Plotting Energy Demand and Production by Day
plt.figure(figsize=(15,10))
energy_daily.plot(
    kind='line', y=["Demand_MW", "Total_Production_MW", "Coal_MW", 
                    "Hydrocarbons_MW", "Water_MW", "Nuclear_MW", 
                    "Wind_MW", "Solar_MW", "Biomass_MW", "Import_MW"]
)
plt.xlabel("Day")
plt.ylabel("MW")
plt.title("Energy Demand and Production by Day")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plotting Energy Demand and Production by Month
plt.figure(figsize=(20,15))
energy_monthly.plot(
    kind='bar', y=["Demand_MW", "Total_Production_MW", "Coal_MW", 
                   "Hydrocarbons_MW", "Water_MW", "Nuclear_MW", 
                   "Wind_MW", "Solar_MW", "Biomass_MW", "Import_MW"]
)
plt.xlabel("Month")
plt.ylabel("MW")
plt.title("Average Energy Demand and Production by Month")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Step 3: Correlation Analysis
# Merge datasets on 'Date'
merged_data = pd.merge(energy_hourly, weather_data, left_index=True, right_index=True)

# Correlation of weather conditions and energy production
correlation = merged_data[[
    'AvgTemp_Celsius', 'AvgWindSpeed_kmh', 'Solar_MW', 'Wind_MW',
    'Coal_MW', 'Hydrocarbons_MW', 'Water_MW', 'Nuclear_MW', 'Biomass_MW', 'Import_MW'
]].corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Step 4: Forecasting Energy Demand and Production
# Setup features and targets for forecasting
X = merged_data[['AvgTemp_Celsius', 'AvgWindSpeed_kmh']]
y = merged_data[[
    'Demand_MW', 'Solar_MW', 'Wind_MW', 'Coal_MW', 'Hydrocarbons_MW',
    'Water_MW', 'Nuclear_MW', 'Biomass_MW', 'Import_MW'
]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train models for each energy source
models = {}
predictions = {}
rmses = {}

for column in y.columns:
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train[column])
    y_pred = model.predict(X_test)
    
    models[column] = model
    predictions[column] = y_pred
    rmses[column] = rmse_percentage(y_test[column], y_pred)
    
    print(f"{column} Forecast RMSE (%): {rmses[column]}")

# Step 5: Visualization of Forecasts vs. Actuals
for column in y.columns:
    plt.figure(figsize=(10,7))
    sns.scatterplot(x=y_test[column], y=predictions[column])
    plt.title(f'{column} Energy Production: Actual vs. Predicted')
    plt.xlabel('Actual Production')
    plt.ylabel('Predicted Production')
    plt.legend(['Predicted'])
    plt.show()

# Step 6: Daily Energy Forecasting for the Next Year with SARIMA
# Increase the order of the non-seasonal components to handle longer forecasts
sarima_model = SARIMAX(
    energy_daily['Demand_MW'],
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 7)
)
sarima_results = sarima_model.fit(disp=True)

# Generate a forecast for the next 365 days
sarima_forecast = sarima_results.forecast(steps=365)

# Plot sarima_forecast on the first subplot
plt.figure(figsize=(20,15))
plt.plot(sarima_forecast)
plt.xlabel("Days")
plt.ylabel("Demand MW")
plt.title("SARIMA Forecast for the next 365 days")
plt.show()

print("SARIMA Forecast for the next 365 days:")
print(sarima_forecast)

# Step 7: Optimization with PuLP
avg_values = {col: np.mean(merged_data[col]) for col in y.columns}

# Constants for Impact Assessment
grid_emission_factor = 0.5
costs_per_kwh = {
    'Solar_MW': 0.05,        # Adjusted cost for Romania
    'Wind_MW': 0.06,         # Adjusted cost for Romania
    'Coal_MW': 0.10,         # Adjusted cost for Romania
    'Hydrocarbons_MW': 0.12, # Adjusted cost for Romania
    'Water_MW': 0.04,        # Adjusted cost for Romania
    'Nuclear_MW': 0.06,      # Adjusted cost for Romania
    'Biomass_MW': 0.08,      # Adjusted cost for Romania
    'Import_MW': 0.09        # Adjusted cost for Romania
}

# Define optimization problem
problem = pulp.LpProblem("Energy_Optimization", pulp.LpMinimize)
supply_vars = {source: pulp.LpVariable(source, lowBound=0) for source in costs_per_kwh.keys()}

# Objective Function
problem += pulp.lpSum([costs_per_kwh[source] * supply_vars[source] for source in supply_vars]), "Total Cost"

# Constraint: Total supply must meet or exceed average daily demand
problem += pulp.lpSum([supply_vars[source] for source in supply_vars]) >= avg_values['Demand_MW'], "Demand_Meeting"

# Solve the problem
problem.solve()

# Check solution status
if pulp.LpStatus[problem.status] == 'Optimal':
    for source in supply_vars:
        print(f"Optimal {source}: {supply_vars[source].varValue} MW")
else:
    print("Optimization did not find a feasible solution.")

# Step 8: Impact Assessment and Visualization
emissions_saved_kg = pulp.lpSum([supply_vars[source].varValue for source in supply_vars]) * grid_emission_factor
cost_with_optimization = pulp.value(problem.objective)
cost_with_gas = avg_values['Demand_MW'] * 0.15
savings = cost_with_gas - cost_with_optimization

print(f"Emissions Saved: {emissions_saved_kg} kg CO2")
print(f"Economic Savings: ${savings}")

# Visualization of optimization results
plt.figure(figsize=(10,6))
plt.bar([source for source in supply_vars], [supply_vars[source].varValue for source in supply_vars], color='skyblue')
plt.title('Optimized Energy Supply Distribution')
plt.xlabel('Energy Source')
plt.ylabel('Energy Supplied (MW)')
plt.show()

plt.figure(figsize=(10,6))
plt.bar(['CO2 Emissions Saved (kg)', 'Economic Savings ($)'], [emissions_saved_kg, savings], color=['red', 'green'])
plt.title('Environmental and Economic Impact of Optimization')
plt.show()
