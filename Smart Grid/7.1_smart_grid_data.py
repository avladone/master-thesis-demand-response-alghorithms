# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:55:20 2024

Given a dataset from a smart grid that includes the hourly power consumption 
of a neighborhood for the year 2023. The dataset is in a CSV file named
smart_grid_data_2023.csv, with two columns: time (in the format YYYY-MM-DD
HH:MM:SS) and consumption (in kilowatt-hours). Your task is to analyze this 
data todetermine the top 5% of hours throughout the year with the highest
power consumption, which will be considered peak hours.

"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('smart_grid_data_2023.csv')

# Convert time to DateTime
df['time'] = pd.to_datetime(df['time'])

# Set time as the index
df.set_index('time', inplace=True)

# Resample to get hourly consumption
hourly_consumption = df.resample('H').sum()

# Calculate the 95th percentile as the peak threshold
peak_treshold = hourly_consumption['consumption'].quantile(0.95)

# Filter for demand peak hours
peak_hours = hourly_consumption[
    hourly_consumption['consumption'] > peak_treshold
]

# Plot the consumption data
plt.figure(figsize=(12,6))
plt.plot(hourly_consumption.index, 
         hourly_consumption['consumption'], 
         label = 'Hourly Consumption')
plt.plot(peak_hours.index,
         peak_hours['consumption'], 'r.',
         label='Peak Hours')

# Plot detailing
plt.title('Power Consumption & Peak Hours')
plt.xlabel('Time')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.tight_layout()

plt.show()