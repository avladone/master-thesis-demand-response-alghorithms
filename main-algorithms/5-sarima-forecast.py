"""
This script performs SARIMA forecasts on daily energy demand and production data.
"""

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the energy dataset with daily resampled data
energy_data = pd.read_csv('energy_daily.csv')
energy_data['Date'] = pd.to_datetime(energy_data['Date'])
energy_data.set_index('Date', inplace=True)

# Function to fit SARIMA model and plot forecast
def sarima_forecast(data, column, title):
    sarima_model = SARIMAX(
        data[column],
        order=(2, 1, 2),  # Increased p and q to 2
        seasonal_order=(1, 1, 1, 7)  # Assuming weekly seasonality
    )

    # Fit the model to the data
    sarima_results = sarima_model.fit(disp=True)

    # Generate a forecast for the next 365 days
    sarima_forecast = sarima_results.get_forecast(steps=365)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=365, freq='D')
    sarima_forecast_df = sarima_forecast.summary_frame()

    # Prevent negative forecasts
    sarima_forecast_df['mean'] = sarima_forecast_df['mean'].clip(lower=0)
    sarima_forecast_df['mean_ci_lower'] = sarima_forecast_df['mean_ci_lower'].clip(lower=0)

    # Plot the results
    plt.figure(figsize=(20, 10))
    plt.plot(data.index, data[column], label='Observed', color='blue')
    plt.plot(forecast_index, sarima_forecast_df['mean'], label='Forecast', color='red')
    plt.fill_between(forecast_index, sarima_forecast_df['mean_ci_lower'], sarima_forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.title(f'SARIMA Forecast of Daily {title} for the Next Year')
    plt.legend()
    plt.show()

    print(f"SARIMA Forecast for the next 365 days for {title}:")
    print(sarima_forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']])

    # Provide a summary of the model's performance
    print(sarima_results.summary())
    return sarima_forecast_df

# Perform SARIMA forecasts for each production type
sarima_forecast(energy_data, 'Demand_MW', 'Energy Demand')
sarima_forecast(energy_data, 'Solar_MW', 'Solar Energy Production')
sarima_forecast(energy_data, 'Wind_MW', 'Wind Energy Production')
sarima_forecast(energy_data, 'Coal_MW', 'Coal Energy Production')
sarima_forecast(energy_data, 'Hydrocarbons_MW', 'Hydrocarbons Energy Production')
sarima_forecast(energy_data, 'Water_MW', 'Water Energy Production')
sarima_forecast(energy_data, 'Nuclear_MW', 'Nuclear Energy Production')
sarima_forecast(energy_data, 'Biomass_MW', 'Biomass Energy Production')
sarima_forecast(energy_data, 'Import_Positive_MW', 'Imported Energy')
sarima_forecast(energy_data, 'Export_Positive_MW', 'Exported Energy')