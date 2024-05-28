# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:25:59 2024

@author: ungur
"""

# # Step 7: Optimization with PuLP
# avg_solar_production_mwh_corrected = np.mean(y_pred_solar_corrected)
# avg_wind_production_mwh = np.mean(y_pred_wind)
# avg_daily_demand_mw = np.mean(y_pred_demand)

# # Constants for Impact Assessment
# grid_emission_factor = 0.5
# solar_cost_per_kwh = 0.05
# wind_cost_per_kwh = 0.07
# natural_gas_cost_per_kwh = 0.15

# # Define optimization problem
# problem = pulp.LpProblem("Energy_Optimization", pulp.LpMinimize)

# # Update the supply variable to use the corrected solar production
# grid_supply = pulp.LpVariable("Grid_Supply", lowBound=0)

# solar_supply = pulp.LpVariable(
#     "Solar_Supply",
#     lowBound=0,
#     upBound=avg_solar_production_mwh_corrected
#     )
# wind_supply = pulp.LpVariable(
#     "Wind_Supply",
#     lowBound=0,
#     upBound=avg_wind_production_mwh
#     )

# # Objective Function
# problem += (
#     solar_cost_per_kwh * solar_supply + 
#     wind_cost_per_kwh * wind_supply + 
#     natural_gas_cost_per_kwh * grid_supply,
#     "Total Cost"
#     )

# problem += (
#     grid_supply + solar_supply + wind_supply >= avg_daily_demand_mw,
#     "Demand_Meeting"
#     )
# problem.solve()

# # Check solution status
# if pulp.LpStatus[problem.status] == 'Optimal':
#     print(f"Optimal grid supply: {grid_supply.varValue} MW")
#     print(f"Optimal solar supply: {solar_supply.varValue} MW")
#     print(f"Optimal wind supply: {wind_supply.varValue} MW")
# else:
#     print("Optimization did not find a feasible solution.")
    
# # Step 8: Impact Assessment and Visualization
# emissions_saved_kg = (
#     solar_supply.varValue + wind_supply.varValue
#     ) * grid_emission_factor
# cost_with_gas = avg_daily_demand_mw * natural_gas_cost_per_kwh
# cost_with_optimization = pulp.value(problem.objective)
# savings = cost_with_gas - cost_with_optimization

# print(f"Emissions Saved: {emissions_saved_kg} kg CO2")
# print(f"Economic Savings: ${savings}")

# # Visualization
# plt.figure(figsize=(10,6))
# plt.bar(
#         ['Grid', 'Solar', 'Wind'],
#         [grid_supply.varValue, solar_supply.varValue, wind_supply.varValue],
#         color=['blue', 'orange', 'green']
#         )
# plt.title('Optimized Energy Supply Distribution')
# plt.xlabel('Energy Source')
# plt.ylabel('Energy Supplied (MW)')
# plt.show()

# plt.figure(figsize=(10,6))
# plt.bar(
#         ['CO2 Emissions Saved (kg)', 'Economic Savings ($)'],
#         [emissions_saved_kg, savings],
#         color=['red', 'green']
#         )
# plt.title('Environmental and Economic Impact of Optimization')
# plt.show()
