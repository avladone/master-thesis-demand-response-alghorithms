# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:42:11 2024

@author: ungur
"""

import pandas as pd
import matplotlib.pyplot as plt

efficiency = 0.15 # Solar panel efficiency
temperature_coefficient = -0.004 # Efficiency change per degree Celsius above 25C 

#Function to calculate solar panel output
def solar_panel_output(irradiance, efficiency, temperature_coefficient, temperature):
    adjusted_efficiency = efficiency * (1 + temperature_coefficient * (temperature - 25))
    output = irradiance * adjusted_efficiency
    return output

#Load the dataset
df = pd.read_csv('solar_data_simulation.csv')

#Convert the timestamp column to datetime format for easier handling
df['new_timestamp'] = pd.to_datetime(df['timestamp'])

# Check for missing values and perform neccesary cleaning 
# This is a simple fill-forward example; more complex strategies might be needed

df.ffill(inplace=True)

print(df)

# Simulate energy output for each data point in the dataset
df['energy_output'] = df.apply(lambda x: solar_panel_output(x['irradiance'], efficiency, temperature_coefficient, x['temperature']), axis=1)

# Plot the energy output over the course of the day

plt.figure(figsize=(12,6))
plt.plot(df['new_timestamp'], df['energy_output']),
plt.xlabel('Time')
plt.ylabel('Energy Output (W)')
plt.title('Simulated Daily Energy Output of a Solar PV System')
plt.grid(True) 
plt.show()

