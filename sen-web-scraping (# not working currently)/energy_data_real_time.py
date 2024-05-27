# -*- coding: utf-8 -*-

import requests
import json
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import os
import time
import urllib.parse

ENERGY_TYPES = ["CARB", "GAZE", "APE", "NUCL", "EOLIAN", "FOTO", "BMASA", "ISPOZ", "SOLD"]

class WebScraper:
    def __init__(self, transelectrica_url, sen_filter_url):
        self.transelectrica_url = transelectrica_url
        self.sen_filter_url = sen_filter_url

    def fetch_timestamp_from_xhr_request(self, url):
        params = {"_": int(round(time.time() * 1000))}
        url_with_params = f"{url}?{urllib.parse.urlencode(params)}"
        response = requests.get(url_with_params)
        if response.status_code == 200:
            data = json.loads(response.text)
            for item in data:
                if "row1_HARTASEN_DATA" in item:
                    timestamp_text = item["row1_HARTASEN_DATA"]
                    try:
                        timestamp = datetime.strptime(timestamp_text, "%y/%m/%d %H:%M:%S")
                        return timestamp
                    except ValueError as e:
                        raise ValueError(f"Invalid timestamp format - {e}")
        else:
            raise ValueError(f"Failed to fetch timestamp - {response.status_code}")

    def fetch_html_content(self, url):
        max_retries = 5
        delay_seconds = 5
        for attempt in range(max_retries):
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for bad status codes
                return response.content
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
                else:
                    print("Maximum retries reached. Exiting...")

    def parse_html_content(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        return soup

    def extract_energy_data(self, soup):
        energy_data = {}
        energy_data["energy_consumption"] = int(soup.select_one("#SEN_Harta_CONS_value").text.strip())
        energy_data["energy_production"] = int(soup.select_one("#SEN_Harta_PROD_value").text.strip())
        for energy_type in ENERGY_TYPES:
            energy_type_div = soup.select_one(f"#SEN_Harta_{energy_type.upper()}_value")
            energy_data[energy_type] = int(energy_type_div.text)
        return energy_data

    def write_csv_header(self, writer):
        writer.writerow(["timestamp", "energy_consumption", "energy_production"] + ENERGY_TYPES)

    def write_csv_data(self, writer, energy_data):
        writer.writerow([energy_data["timestamp"], energy_data["energy_consumption"], energy_data["energy_production"]] + [energy_data[energy_type] for energy_type in ENERGY_TYPES])

    def write_to_csv(self, energy_data, file_path):
        if os.path.exists(file_path):
            with open(file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                self.write_csv_data(writer, energy_data)
        else:
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                self.write_csv_header(writer)
                self.write_csv_data(writer, energy_data)

def main():
    transelectrica_url = "https://www.transelectrica.ro/widget/web/tel/sen-harta/-/harta_WAR_SENOperareHartaportlet"
    sen_filter_url = "https://www.transelectrica.ro/sen-filter"
    file_path = os.path.join(os.getcwd(), "energy_data_real_time.csv")

    scraper = WebScraper(transelectrica_url, sen_filter_url)
    html_content = scraper.fetch_html_content(transelectrica_url)
    soup = scraper.parse_html_content(html_content)
    timestamp = scraper.fetch_timestamp_from_xhr_request(sen_filter_url)
    energy_data = scraper.extract_energy_data(soup)
    energy_data["timestamp"] = timestamp
    scraper.write_to_csv(energy_data, file_path)

if __name__ == "__main__":
    main()