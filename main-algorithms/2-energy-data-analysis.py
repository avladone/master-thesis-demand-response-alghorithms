"""
This file takes the energy data that was cleaned in the previous file
1-energy-data-fetching.py and resamples it to hourly, daily and monthly values.
It also does preliminary analysis, by creating plots in order to visualize the 
data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utilities import plot_line_with_rolling_mean

# Load energy dataset
energy_data = pd.read_csv('data/outputs/cleaned_Grafic_SEN.csv')
energy_data['Date'] = pd.to_datetime(energy_data['Date'])
energy_data.set_index('Date', inplace=True)

# Identify and split import and export values
energy_data['Import_Positive_MW'] = energy_data['Import_MW'].apply(lambda x: x if x > 0 else 0)
energy_data['Export_Positive_MW'] = energy_data['Import_MW'].apply(lambda x: -x if x < 0 else 0)

# Resampling data to hourly, daily, and monthly values
energy_hourly = energy_data.resample("H").mean()
energy_daily = energy_data.resample("D").mean()
energy_monthly = energy_data.resample("M").mean()

# Save resampled data to CSV files
energy_hourly.to_csv('data/outputs/energy_hourly.csv')
energy_daily.to_csv('data/outputs/energy_daily.csv')
energy_monthly.to_csv('data/outputs/energy_monthly.csv')

# Preliminary Data Analysis
print(energy_data.head())
print("\nSummary Statistics (Hourly):")
print(energy_hourly.describe())

# Rolling Mean for Smoothing
energy_hourly_rolling = energy_hourly.rolling(window=7).mean()
energy_daily_rolling = energy_daily.rolling(window=7).mean()

# Hourly Graphs
plot_line_with_rolling_mean(
    energy_hourly_rolling,
    ['Demand_MW', 'Total_Production_MW', 'Export_Positive_MW'],
    "Energy Demand, Total Production, and Export by Hour (7-Day Rolling Mean)",
    "Hour",
    "MW"
)

plot_line_with_rolling_mean(
    energy_hourly_rolling,
    ['Coal_MW', 'Hydrocarbons_MW', 'Water_MW', 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW', 'Import_Positive_MW'],
    "Energy Production Types and Import by Hour (7-Day Rolling Mean)",
    "Hour",
    "MW"
)

# Daily Graphs
plot_line_with_rolling_mean(
    energy_daily_rolling,
    ['Demand_MW', 'Total_Production_MW', 'Export_Positive_MW'],
    "Energy Demand, Total Production, and Export by Day (7-Day Rolling Mean)",
    "Day",
    "MW"
)

plot_line_with_rolling_mean(
    energy_daily_rolling,
    ['Coal_MW', 'Hydrocarbons_MW', 'Water_MW', 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW', 'Import_Positive_MW'],
    "Energy Production Types and Import by Day (7-Day Rolling Mean)",
    "Day",
    "MW"
)

# Monthly Graphs
fig, ax1 = plt.subplots(figsize=(20, 15))
energy_monthly[['Demand_MW', 'Total_Production_MW', 'Export_Positive_MW']].plot(kind='bar', ax=ax1)
ax1.set_xlabel("Month")
ax1.set_ylabel("MW")
ax1.set_title("Energy Demand, Total Production, and Export by Month")
months = mdates.MonthLocator()  # every month
month_fmt = mdates.DateFormatter('%b')
ax1.xaxis.set_major_locator(months)
ax1.xaxis.set_major_formatter(month_fmt)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True)
plt.show()

# Graph 2: All Production Types + Import
fig, ax1 = plt.subplots(figsize=(20, 15))
energy_monthly[['Coal_MW', 'Hydrocarbons_MW', 'Water_MW', 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW', 'Import_Positive_MW']].plot(kind='bar', stacked=True, ax=ax1)
ax1.set_xlabel("Month")
ax1.set_ylabel("MW")
ax1.set_title("Energy Production Types and Import by Month")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Format x-axis timestamps to show abbreviated month names
months = mdates.MonthLocator()  # every month
month_fmt = mdates.DateFormatter('%b')
ax1.xaxis.set_major_locator(months)
ax1.xaxis.set_major_formatter(month_fmt)
ax1.grid(True)
plt.show()