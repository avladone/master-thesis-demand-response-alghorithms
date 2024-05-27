# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:37:41 2024

@author: ungur
"""
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value

# Wind turbine parameters
CUT_IN_SPEED = 3.5 # Minimum operational wind speed (m/s)
RATED_SPEED = 15.0 # Optimal operational wind speed (m/s)
CUT_OUT_SPEED = 25.0 # Maximum operational wind speed (m/s)
MAXIMUM_POWER_OUTPUT = 1500 # Maximum power output (kW)

# Maintenance hours
MAINTENANCE_HOURS = [20, 21, 22, 23]

# Load the CSV file into a DataFrame
df = pd.read_csv('hypothetical_wind_turbine_data.csv')

# Define wind speed profile from the CSV file
wind_speed_profile = df['Wind Speed (m/s)']

# Creating a linear programming problem
lp_problem = LpProblem('Maximize_Energy_Output', LpMaximize)

# Creating decision variables for power output
power_output = LpVariable.dicts("PowerOutput", 
                                range(24), 
                                lowBound = 0, 
                                upBound = MAXIMUM_POWER_OUTPUT, 
                                cat = 'Continous')

# Defining the objective function
lp_problem += lpSum([power_output[hour] for hour in range(24)])

# Adding constraints for wind speeds and maintenance hours
# Loop through each hour to set constraints 
for hour in range(24):
    wind_speed = wind_speed_profile[hour]
    lp_problem += power_output[hour] <= (wind_speed >= CUT_IN_SPEED)*(wind_speed <= CUT_OUT_SPEED)  * MAXIMUM_POWER_OUTPUT
        
    if hour in MAINTENANCE_HOURS:
        lp_problem += power_output[hour] == 0
    
    lp_problem += power_output[hour] <= MAXIMUM_POWER_OUTPUT
    
# Solving the linear programming problem
lp_problem.solve()

# Extracting the optimal schedule and power output
optimal_output = [value(power_output[hour]) for hour in range(24)]

# Calculating the total optimized power output and plotting the results
total_optimized_output = sum(optimal_output)

# Plotting the results
plt.figure(figsize=(10,6))
plt.plot(optimal_output, drawstyle='steps-post', marker='o', linestyle='-', 
         color='green')
plt.fill_between(range(24), optimal_output, step='post', alpha=0.4, 
                 color='green')
plt.title('Optimal Power Output over 24 Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Power Output (kW)')
plt.grid(axis='y', linestyle='--')
plt.show()

print("Status: ", LpStatus[problem.status])
print(f"Optimal Power Output Schedule: {optimal_output}")
print(f"Total Optimized Power Output: {total_optimized_output} kW")