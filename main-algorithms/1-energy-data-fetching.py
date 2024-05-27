"""
The first Python file that is part of my project reads the preliminary energy 
data from the National Energy System with the end purpose of creating a .csv file 
that can be used as inputs in the next files.
"""

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Load the data
file_path = 'Grafic_SEN.xlsx'
sheet_data = pd.read_excel(file_path, sheet_name='Grafic SEN')

# Convert 'Data' column to datetime format
sheet_data['Data'] = pd.to_datetime(sheet_data['Data'], format='%d-%m-%Y %H:%M:%S')

# Remove non-numeric characters and convert to numeric dtype
def clean_numeric_columns(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column].astype(str).str.replace(r'[^0-9.-]', ''), errors='coerce')
    return df

# Define columns to clean and interpolate
columns_to_interpolate = ['Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 
                          'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 
                          'Biomasa[MW]', 'Sold[MW]']

sheet_data = clean_numeric_columns(sheet_data, columns_to_interpolate)

# Remove negative values except for 'Sold[MW]'
for column in columns_to_interpolate:
    if column != 'Sold[MW]':
        sheet_data[column] = sheet_data[column].apply(lambda x: x if x >= 0 else np.nan)

# Sort the data by 'Data' column from oldest to newest
sheet_data = sheet_data.sort_values(by='Data').reset_index(drop=True)

# Interpolation function to fill gaps
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

# Fill gaps in the data
filled_data = fill_gaps(sheet_data, 'Data', columns_to_interpolate)

# Constrain interpolated values to the range of original data
for column in columns_to_interpolate:
    min_val = sheet_data[column].min()
    max_val = sheet_data[column].max()
    filled_data[column] = filled_data[column].clip(lower=min_val, upper=max_val)

# Remove the last two rows and the "Medie Consum[MW]" column
filled_data = filled_data.dropna(subset=['Data']).iloc[:-2]
filled_data = filled_data.drop(columns=['Medie Consum[MW]'])

# Rename the columns as specified
filled_data.columns = ['Date', 'Demand_MW', 'Total_Production_MW', 'Coal_MW', 'Hydrocarbons_MW',
                      'Water_MW', 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW', 'Import_MW']

# Save the resulting file as a .csv
csv_file_path = 'cleaned_Grafic_SEN.csv'
filled_data.to_csv(csv_file_path, index=False)

# Provide the path to the saved file
print("Cleaned CSV file saved to:", csv_file_path)
