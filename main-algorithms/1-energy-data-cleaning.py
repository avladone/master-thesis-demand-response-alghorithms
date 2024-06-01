"""
The first Python file that is part of my project reads the preliminary energy 
data from the National Energy System with the end purpose of creating a .csv file 
that can be used as inputs in the next files.
"""

import pandas as pd
import numpy as np
from utilities import clean_numeric_columns, fill_gaps

# Load the data
file_path = 'data/inputs/Grafic_SEN.xlsx'
sheet_data = pd.read_excel(file_path, sheet_name='Grafic SEN')

# Convert 'Data' column to datetime format
sheet_data['Data'] = pd.to_datetime(sheet_data['Data'], format='%d-%m-%Y %H:%M:%S')

# Define columns to clean and interpolate
columns_to_interpolate = [
    'Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 
    'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 
    'Biomasa[MW]', 'Sold[MW]'
]

sheet_data = clean_numeric_columns(sheet_data, columns_to_interpolate)

# Remove negative values except for 'Sold[MW]'
for column in columns_to_interpolate:
    if column != 'Sold[MW]':
        sheet_data[column] = sheet_data[column].apply(lambda x: x if x >= 0 else np.nan)

# Sort the data by 'Data' column from oldest to newest
sheet_data = sheet_data.sort_values(by='Data').reset_index(drop=True)

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
filled_data.columns = [
    'Date', 'Demand_MW', 'Total_Production_MW', 'Coal_MW', 'Hydrocarbons_MW',
    'Water_MW', 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW', 'Import_MW'
]

# Save the resulting file as a .csv
csv_file_path = 'data/outputs/cleaned_Grafic_SEN.csv'
filled_data.to_csv(csv_file_path, index=False)

# Provide the path to the saved file
print("Cleaned CSV file saved to:", csv_file_path)