import requests
import pandas as pd
import time
from datetime import datetime, timedelta

API_KEY = '2025c4940c2133be0650d1cf324d7bd0'  # Replace with your actual OpenWeatherMap API key

# List of coordinates for multiple locations in Romania
locations = [
    {'lat': 44.4268, 'lon': 26.1025},  # Bucharest
    {'lat': 46.7712, 'lon': 23.6236},  # Cluj-Napoca
    {'lat': 45.7489, 'lon': 21.2087},  # Timișoara
    {'lat': 44.3302, 'lon': 23.7949},  # Craiova
    {'lat': 44.1598, 'lon': 28.6348},  # Constanta
    {'lat': 47.0722, 'lon': 21.9214},  # Oradea
    {'lat': 46.5425, 'lon': 24.5575},  # Targul Mures
    {'lat': 47.1585, 'lon': 27.6014},  # Iasi
    {'lat': 46.6407, 'lon': 27.7276},  # Vaslui
]

# Function to fetch data for a single location and timestamp
def fetch_data(lat, lon, timestamp):
    url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
    params = {
        'lat': lat,
        'lon': lon,
        'dt': timestamp,
        'appid': API_KEY,
        'units': 'metric'
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'hourly' in data:
        return data['hourly']
    else:
        print(f"Error fetching data for {lat}, {lon} at {datetime.utcfromtimestamp(timestamp)}: {data}")
        return []

# Define the start and end dates for 2023
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31, 23)  # 23:00 on Dec 31, 2023

# Create a DataFrame to store the results
df_combined = pd.DataFrame(columns=['datetime', 'temperature', 'wind_speed', 'solar_radiation'])

# Loop through each hour of the year
current_date = start_date
while current_date <= end_date:
    timestamp = int(current_date.timestamp())
    
    # Initialize lists to store data from all locations
    temps = []
    wind_speeds = []
    solar_radiations = []
    
    # Fetch data for each location
    for location in locations:
        hourly_data = fetch_data(location['lat'], location['lon'], timestamp)
        for hour in hourly_data:
            temps.append(hour['temp'])
            wind_speeds.append(hour['wind_speed'])
            solar_radiations.append(hour.get('solar_radiation', 0))  # Default to 0 if not available
    
    # If data was fetched successfully from at least one location, calculate the averages
    if temps and wind_speeds and solar_radiations:
        avg_temp = sum(temps) / len(temps)
        avg_wind_speed = sum(wind_speeds) / len(wind_speeds)
        avg_solar_radiation = sum(solar_radiations) / len(solar_radiations)
    
        # Append to the DataFrame
        df_combined = df_combined.append({
            'datetime': current_date,
            'temperature': avg_temp,
            'wind_speed': avg_wind_speed,
            'solar_radiation': avg_solar_radiation
        }, ignore_index=True)
    else:
        print(f"No valid data for {current_date}")
    
    # Move to the next hour
    current_date += timedelta(hours=1)
    
    # Sleep to avoid hitting API rate limits
    time.sleep(1)

# Save to CSV
df_combined.to_csv('average_weather_romania_2023.csv', index=False)

print("Data collection complete. Saved to average_weather_romania_2023.csv.")
