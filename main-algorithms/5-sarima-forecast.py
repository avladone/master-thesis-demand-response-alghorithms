# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:23:56 2024

@author: ungur
"""

# # Step 6: Daily Energy Forecasting for the next Year with SARIMA

# # Increase the order of the non-seasonal components to handle longer forecasts
# sarima_model = SARIMAX(
#     energy_daily['Demand_MW'],
#     order=(2, 1, 2),  # Increased p and q to 2
#     seasonal_order=(1, 1, 1, 7)
#     )

# # Fit the model to the data
# sarima_results = sarima_model.fit(disp=True)

# # Generate a forecast for the next 365 days
# sarima_forecast = sarima_results.forecast(steps=365)

# # Plot sarima_forecast on the first subplot
# plt.figure(figsize=(20,15))
# sarima_forecast.plot(
#     kind='line',
#     x = 'predicted_mean',
#     y = 'index')

# plt.show()

# print("SARIMA Forecast for the next 365 days:")
# print(sarima_forecast)