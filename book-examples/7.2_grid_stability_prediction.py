# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:24:41 2024

You are provided with a dataset grid_data.csv that contains hourly electricity 
demand (in MW) and the hourly output from renewable sources (in MW) for a 
region. Your task is to develop a Python program that uses machine learning to
predict the grid stability index, defined as the difference between electricity
demand and renewable output, for the next 24 hours. This index helps in 
understanding potential stress on the grid due to variability in renewable 
output.

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset into a DataFrame
df = pd.read_csv('grid_data.csv')

# Step 2: Exploratory Data Analysis
# Ensure 'DateTime' is a datetime type
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Plotting demand and renewable output
plt.figure(figsize=(10,6))
plt.plot(df['DateTime'], df['Electricity_Demand_MW'],
         label = 'Electricity Demand')
plt.plot(df['DateTime'], df['Renewable_Output_MW'],
         label = 'Renewable Output')

# Formatting Date on x-axis
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M')
plt.gca().xaxis.set_major_formatter(date_format)

# Detailing the plot and displaying it
plt.title('Demand vs. Output Over Time')
plt.xlabel('DateTime')
plt.ylabel('MW')
plt.legend()
plt.show()

# Step 3: Create ‘grid_stability_index’ Feature
# Calculate grid stability index
df['grid_stability_index'] = df['Electricity_Demand_MW']\
    - df['Renewable_Output_MW']

# Step 4: Split the Dataset
# Define features and target variable
X = df[['Electricity_Demand_MW', 'Renewable_Output_MW']] # Features
y = df['grid_stability_index'] # Target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Model and Predict with RandomForestRegressor
# Initialize the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,
                              random_state=42)

# Train the model 
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Step 7: Visualize Actual vs. Predicted
plt.figure(figsize=(10,6))

plt.scatter(y_test.index, y_test, label='Actual', 
             color='blue', alpha=0.5, s=10)
plt.scatter(y_test.index, predictions, label='Predicted',
             color='red', alpha=0.3, s=10)

plt.title('Actual vs. Predicted\nGrid Stability Index')
plt.xlabel('Index')
plt.ylabel('Grid Stability Index')
plt.legend()
plt.show()
