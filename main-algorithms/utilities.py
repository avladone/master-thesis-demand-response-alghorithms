"""
This script contains various functions that are used throughout the other scripts
of this project, in order to have a cleaner and a more modular code.
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Utility Functions

# Function to clean numeric columns
def clean_numeric_columns(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column].astype(str).str.replace(r'[^0-9.-]', ''), errors='coerce')
    return df

# Function to fill gaps in the data
def fill_gaps(data, time_column, columns_to_interpolate, freq='10T'):
    result = []
    for i in range(len(data) - 1):
        current_row = data.iloc[i]
        next_row = data.iloc[i + 1]
        result.append(current_row)
        
        time_diff = next_row[time_column] - current_row[time_column]
        if time_diff > pd.Timedelta(freq):
            num_missing = int(time_diff / pd.Timedelta(freq)) - 1
            for j in range(1, num_missing + 1):
                new_row = current_row.copy()
                new_row[time_column] = current_row[time_column] + j * pd.Timedelta(freq)
                for col in columns_to_interpolate:
                    new_row[col] = np.nan
                result.append(new_row)
    
    result.append(data.iloc[-1])
    result_df = pd.DataFrame(result)
    
    # Convert relevant columns to numeric dtype
    for column in columns_to_interpolate:
        result_df[column] = pd.to_numeric(result_df[column], errors='coerce')
    
    # Interpolate and fill forward and backward to ensure no NaNs remain
    result_df[columns_to_interpolate] = result_df[columns_to_interpolate].interpolate().ffill().bfill()
    
    return result_df

# Function to create line plots with rolling mean
def plot_line_with_rolling_mean(data, columns, title, xlabel, ylabel, window=7, figsize=(15, 10), legend_loc='upper left'):
    rolling_data = data[columns].rolling(window=window).mean()
    plt.figure(figsize=figsize)
    rolling_data.plot(kind='line')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=legend_loc)
    plt.grid(True)
    plt.show()

# Function to create bar plots
def plot_bar(data, columns, title, xlabel, ylabel, figsize=(20, 15), legend_loc='upper left'):
    fig, ax1 = plt.subplots(figsize=figsize)
    data[columns].plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc=legend_loc)
    plt.grid(True)
    plt.show()

# Function to create scatter plots
def plot_scatter(data, x_col, y_col, title, x_label, y_label, color='blue', alpha=0.5, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.scatterplot(x=data[x_col], y=data[y_col], color=color, alpha=alpha)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

# Function to create correlation heatmap
def plot_correlation_heatmap(data, title, figsize=(10, 8), cmap='coolwarm', linewidths=0.5):
    correlation = data.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation, annot=True, cmap=cmap, linewidths=linewidths)
    plt.title(title)
    plt.show()

# Function to calculate RMSE percentage
def rmse_percentage(true_values, predicted_values):
    rmse = sqrt(mean_squared_error(true_values, predicted_values))
    average = np.mean(true_values)
    return (rmse / average) * 100

# Function to calculate relative error
def relative_error(true_values, predicted_values):
    return (abs((true_values - predicted_values) / true_values) * 100)

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(df, title, xlabel='Time', ylabel='Values', figsize=(10, 7)):
    plt.figure(figsize=figsize)
    plt.plot(df['Actual'].values, label='Actual')
    plt.plot(df['Predicted'].values, label='Predicted')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14) 
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# Function to plot relative error
def plot_relative_error(df, title, xlabel='Time', ylabel='Relative Error (%)', figsize=(10, 7)):
    plt.figure(figsize=figsize)
    plt.plot(df['Relative_Error'].values)
    plt.yscale('log')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def plot_optimized_energy_distribution(optimal_supplies):
    plt.figure(figsize=(10, 6))
    plt.bar(
        optimal_supplies.keys(),
        optimal_supplies.values(),
        color=['brown', 'orange', 'blue', 'red', 'green', 'purple', 'yellow', 'green']
    )
    plt.title('Optimized Energy Supply Distribution')
    plt.xlabel('Energy Source')
    plt.ylabel('Energy Supplied (MW)')
    plt.grid(True)
    plt.show()

def plot_impact_assessment(emissions_saved_kg, savings):
    plt.figure(figsize=(10, 6))
    plt.bar(
        ['CO2 Emissions Saved (kg)', 'Economic Savings (RON)'],
        [emissions_saved_kg, savings],
        color=['red', 'green']
    )
    plt.title('Environmental and Economic Impact of Optimization')
    plt.grid(True)
    plt.show()