"""
This script performs grid optimization using the predicted energy data.
"""

import pandas as pd
import numpy as np
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, value as lp_value
from utilities import plot_optimized_energy_distribution, plot_impact_assessment

# Load predicted energy data
predicted_data = pd.read_csv('data/outputs/energy_predictions.csv')

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

# Constants for Impact Assessment (Adjusted for Romania)
# Convert g CO2/kWh to kg CO2/kWh
emission_factors = {
    'Coal': 935 / 1000,
    'Hydrocarbons': 500 / 1000,
    'Hydropower': 3 / 1000,
    'Nuclear': 15 / 1000,
    'Solar': 45 / 1000,
    'Wind': 15 / 1000,
    'Biomass': 75 / 1000,
    'Import': 227 / 1000  # General average of all the other factors
}
solar_cost_per_kwh = 0.3  # RON
wind_cost_per_kwh = 0.25  # RON
coal_cost_per_kwh = 0.45  # RON
hydrocarbons_cost_per_kwh = 0.4  # RON
water_cost_per_kwh = 0.15  # RON
nuclear_cost_per_kwh = 0.2  # RON
biomass_cost_per_kwh = 0.35  # RON
import_cost_per_kwh = 0.52  # RON
export_income_per_kwh = 0.68  # RON

# Define optimization problem
problem = LpProblem("Energy_Optimization", LpMinimize)

# Define variables for the optimization problem
coal_supply = LpVariable("Coal_Supply", lowBound=0, upBound=avg_predicted_coal)
hydrocarbons_supply = LpVariable("Hydrocarbons_Supply", lowBound=0, upBound=avg_predicted_hydrocarbons)
water_supply = LpVariable("Water_Supply", lowBound=0, upBound=avg_predicted_water)
nuclear_supply = LpVariable("Nuclear_Supply", lowBound=0, upBound=avg_predicted_nuclear)
biomass_supply = LpVariable("Biomass_Supply", lowBound=0, upBound=avg_predicted_biomass)
import_supply = LpVariable("Import_Supply", lowBound=0, upBound=avg_predicted_import)
export_supply = LpVariable("Export_Supply", lowBound=0, upBound=avg_predicted_export)
solar_supply = LpVariable("Solar_Supply", lowBound=0, upBound=avg_predicted_solar)
wind_supply = LpVariable("Wind_Supply", lowBound=0, upBound=avg_predicted_wind)

# Emission variable (weighted for minimization)
emissions = (
    coal_supply * emission_factors['Coal'] +
    hydrocarbons_supply * emission_factors['Hydrocarbons'] +
    water_supply * emission_factors['Hydropower'] +
    nuclear_supply * emission_factors['Nuclear'] +
    biomass_supply * emission_factors['Biomass'] +
    import_supply * emission_factors['Import'] +
    solar_supply * emission_factors['Solar'] +
    wind_supply * emission_factors['Wind']
)

# Objective Function: Minimize total cost considering export income and emissions
cost_weight = 0  # Adjust weight as needed
emission_weight = 1  # Adjust weight as needed
problem += (
    cost_weight * (
        coal_cost_per_kwh * coal_supply +
        hydrocarbons_cost_per_kwh * hydrocarbons_supply +
        water_cost_per_kwh * water_supply +
        nuclear_cost_per_kwh * nuclear_supply +
        biomass_cost_per_kwh * biomass_supply +
        import_cost_per_kwh * import_supply +
        solar_cost_per_kwh * solar_supply +
        wind_cost_per_kwh * wind_supply -
        export_income_per_kwh * export_supply
    ) + emission_weight * emissions,
    "Total Cost and Emissions"
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
if LpStatus[problem.status] == 'Optimal':
    optimal_supplies = {
        'Coal': coal_supply.varValue,
        'Hydrocarbons': hydrocarbons_supply.varValue,
        'Water': water_supply.varValue,
        'Nuclear': nuclear_supply.varValue,
        'Biomass': biomass_supply.varValue,
        'Import': import_supply.varValue,
        'Export': export_supply.varValue,
        'Solar': solar_supply.varValue,
        'Wind': wind_supply.varValue
    }

    for supply, supply_value in optimal_supplies.items():
        print(f"Optimal {supply.lower()} supply: {supply_value:.2f} MW")
else:
    print("Optimization did not find a feasible solution.")

# Impact Assessment
# Calculate emissions from polluting sources in the optimized scenario
emissions_optimized = (
    coal_supply.varValue * emission_factors['Coal'] +
    hydrocarbons_supply.varValue * emission_factors['Hydrocarbons'] +
    water_supply.varValue * emission_factors['Hydropower'] +
    nuclear_supply.varValue * emission_factors['Nuclear'] +
    biomass_supply.varValue * emission_factors['Biomass'] +
    import_supply.varValue * emission_factors['Import'] +
    solar_supply.varValue * emission_factors['Solar'] +
    wind_supply.varValue * emission_factors['Wind']
)

# Calculate emissions from polluting sources in the average scenario
emissions_average = (
    avg_predicted_coal * emission_factors['Coal'] +
    avg_predicted_hydrocarbons * emission_factors['Hydrocarbons'] +
    avg_predicted_water * emission_factors['Hydropower'] +
    avg_predicted_nuclear * emission_factors['Nuclear'] +
    avg_predicted_biomass * emission_factors['Biomass'] +
    avg_predicted_import * emission_factors['Import'] +
    avg_predicted_solar * emission_factors['Solar'] +
    avg_predicted_wind * emission_factors['Wind']
)

# Emissions saved
emissions_saved_kg = emissions_average - emissions_optimized

cost_with_grid_only = (
    avg_predicted_coal * coal_cost_per_kwh +
    avg_predicted_hydrocarbons * hydrocarbons_cost_per_kwh +
    avg_predicted_water * water_cost_per_kwh +
    avg_predicted_nuclear * nuclear_cost_per_kwh +
    avg_predicted_biomass * biomass_cost_per_kwh +
    avg_predicted_import * import_cost_per_kwh +
    avg_predicted_solar * solar_cost_per_kwh +
    avg_predicted_wind * wind_cost_per_kwh
)
cost_with_optimization = lp_value(problem.objective)
savings = cost_with_grid_only - cost_with_optimization

print(f"Emissions Saved: {emissions_saved_kg:.2f} kg CO2")
print(f"Economic Savings: {savings:.2f} RON")

# Visualization
plot_optimized_energy_distribution(optimal_supplies)
plot_impact_assessment(emissions_saved_kg, savings)

# Create a DataFrame with the results
optimal_production_df = pd.DataFrame.from_dict(optimal_supplies, orient='index', columns=['Optimal Production'])

# Save the results to a CSV file
optimal_production_df.to_csv('data/outputs/optimal_production.csv')

print("Optimal production saved to data/outputs/optimal_production.csv")
