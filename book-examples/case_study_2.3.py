# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:10:43 2024

@author: ungur
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
solar_data = pd.read_csv('solar_panel_efficiency_data.csv')

#Visualize the relationship between the irradiance, temperature and efficiency

plt.figure(figsize=(12, 6))
sns.scatterplot(data=solar_data, x='temperature', y='efficiency', 
                hue='efficiency', palette='coolwarm', alpha=0.6)

plt.title('Solar Efficiency: Temperature & Irradiance Impact', fontsize=18)
plt.xlabel('Temperature (C)', fontsize=14)
plt.ylabel('Irradiance', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Efficiency', fontsize=12,title_fontsize='13')
plt.show()

#Time series plot of solar panel output
plt.figure(figsize=(12,6))
solar_data['date'] = pd.to_datetime(solar_data['date'])
solar_data.set_index('date', inplace=True)

# Plotting the 'panel_output' as a time series
solar_data['panel_output'].plot(
    title='Solar Panel Output',
    color='orange',
    fontsize=14
    )
plt.xlabel('Date', fontsize=14)
plt.ylabel('Panel Output (kWh)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Solar Panel Output Over Time", fontsize=13)
plt.show()

