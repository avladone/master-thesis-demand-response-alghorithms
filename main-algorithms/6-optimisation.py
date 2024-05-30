"""
This script performs grid optimization using the predicted energy data.
"""

# Step 1: Import Libraries and Load Datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp

# Load predicted energy data
predicted_data = pd.read_csv('energy_predictions_gbm.csv')

# Calculate average predictions
avg_predicted_demand = np.mean(predicted_data['Predicted_Demand_MW'])
avg_predicted_solar = np.mean(predicted_data['Predicted_Solar_MW'])
avg_predicted_wind = np.mean(predicted_data['Predicted_Wind_MW'])
avg_predicted_coal = np.mean(predicted_data['Predicted_Coal_MW'])
avg_predicted_hydrocarbons = np.mean(predicted_data['Predicted_Hydrocarbons_MW'])
avg_predicted_water = np.mean(predicted_data['Predicted_Water_MW'])
avg_predicted_nuclear = np.mean(predicted_data['Predicted_Nuclear_MW'])
avg_predicted_biomass = np.mean(predicted_data['Predicted_Biomass_MW'])
avg_predicted_import = np.mean(predicted_data['Predicted_Import_MW'])
avg_predicted_export = np.mean(predicted_data['Predicted_Export_MW'])

# Constants for Impact Assessment
grid_emission_factor = 0.5
coal_cost_per_kwh = 0.10
hydrocarbons_cost_per_kwh = 0.15
water_cost_per_kwh = 0.05
nuclear_cost_per_kwh = 0.07
biomass_cost_per_kwh = 0.08
import_cost_per_kwh = 0.20
export_income_per_kwh = 0.15

# Define optimization problem
problem = pulp.LpProblem("Energy_Optimization", pulp.LpMinimize)

# Define variables for the optimization problem
coal_supply = pulp.LpVariable("Coal_Supply", lowBound=0, upBound=avg_predicted_coal)
hydrocarbons_supply = pulp.LpVariable("Hydrocarbons_Supply", lowBound=0, upBound=avg_predicted_hydrocarbons)
water_supply = pulp.LpVariable("Water_Supply", lowBound=0, upBound=avg_predicted_water)
nuclear_supply = pulp.LpVariable("Nuclear_Supply", lowBound=0, upBound=avg_predicted_nuclear)
biomass_supply = pulp.LpVariable("Biomass_Supply", lowBound=0, upBound=avg_predicted_biomass)
import_supply = pulp.LpVariable("Import_Supply", lowBound=0, upBound=avg_predicted_import)
export_supply = pulp.LpVariable("Export_Supply", lowBound=0, upBound=avg_predicted_export)
solar_supply = pulp.LpVariable("Solar_Supply", lowBound=0, upBound=avg_predicted_solar)
wind_supply = pulp.LpVariable("Wind_Supply", lowBound=0, upBound=avg_predicted_wind)

# Objective Function: Minimize total cost considering export income
problem += (
    coal_cost_per_kwh * coal_supply +
    hydrocarbons_cost_per_kwh * hydrocarbons_supply +
    water_cost_per_kwh * water_supply +
    nuclear_cost_per_kwh * nuclear_supply +
    biomass_cost_per_kwh * biomass_supply +
    import_cost_per_kwh * import_supply -
    export_income_per_kwh * export_supply,
    "Total Cost"
)

# Constraint: Meet the average daily demand
problem += (
    coal_supply + hydrocarbons_supply + water_supply + nuclear_supply + biomass_supply + 
    import_supply + solar_supply + wind_supply - export_supply >= avg_predicted_demand,
    "Demand_Meeting"
)

# Solve the optimization problem
problem.solve()

# Check solution status
if pulp.LpStatus[problem.status] == 'Optimal':
    print(f"Optimal coal supply: {coal_supply.varValue} MW")
    print(f"Optimal hydrocarbons supply: {hydrocarbons_supply.varValue} MW")
    print(f"Optimal water supply: {water_supply.varValue} MW")
    print(f"Optimal nuclear supply: {nuclear_supply.varValue} MW")
    print(f"Optimal biomass supply: {biomass_supply.varValue} MW")
    print(f"Optimal import supply: {import_supply.varValue} MW")
    print(f"Optimal export supply: {export_supply.varValue} MW")
    print(f"Optimal solar supply: {solar_supply.varValue} MW")
    print(f"Optimal wind supply: {wind_supply.varValue} MW")
else:
    print("Optimization did not find a feasible solution.")
    
# Step 8: Impact Assessment and Visualization
emissions_saved_kg = (
    solar_supply.varValue + wind_supply.varValue
) * grid_emission_factor
cost_with_grid_only = avg_predicted_demand * import_cost_per_kwh
cost_with_optimization = pulp.value(problem.objective)
savings = cost_with_grid_only - cost_with_optimization

print(f"Emissions Saved: {emissions_saved_kg} kg CO2")
print(f"Economic Savings: ${savings}")

# Visualization
plt.figure(figsize=(10,6))
plt.bar(
    ['Coal', 'Hydrocarbons', 'Water', 'Nuclear', 'Biomass', 'Import', 'Solar', 'Wind'],
    [coal_supply.varValue, hydrocarbons_supply.varValue, water_supply.varValue, nuclear_supply.varValue, 
     biomass_supply.varValue, import_supply.varValue, solar_supply.varValue, wind_supply.varValue],
    color=['brown', 'orange', 'blue', 'red', 'green', 'purple', 'yellow', 'green']
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