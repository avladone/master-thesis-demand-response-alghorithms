# -*- coding: utf-8 -*-
"""
Web scraping script that gets the energy data for a month period from the Transelectrica site.
"""

# Importing neccesary libraries
import time
import pandas as pd
import requests

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

MONTHS = ["ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie", "august", "septembrie", "octombrie", "noiembrie", "decembrie"]

# Initializing the web scraping
service = Service(executable_path='chromedriver.exe')

options = Options()
options.add_argument("--disable-page-reload")

driver = webdriver.Chrome(service=service, options = options)
driver.get("https://www.transelectrica.ro/widget/web/tel/sen-grafic/-/SENGrafic_WAR_SENGraficportlet")


# Wait for the page to load
wait = WebDriverWait(driver, 10)

# Find the search button
search_button = wait.until(EC.presence_of_element_located((By.ID, "_SENGrafic_WAR_SENGraficportlet_submit_button")))


# Select the starting month drop-down list
start_month_select = wait.until(EC.presence_of_element_located((By.ID, "_SENGrafic_WAR_SENGraficportlet_start_month")))
ActionChains(driver).move_to_element(start_month_select).click().perform()

# Create a Select object with the start_month_select WebElement
select = Select(start_month_select)

# Get the current month value
current_month_value = start_month_select.get_attribute("value")

# Wait for the options to be available
options = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//select[@id='_SENGrafic_WAR_SENGraficportlet_start_month']/option")))

# Select the previous month
if int(start_month_select.get_attribute("value")) == 11:
    select.select_by_visible_text('Ianuarie')
else:
    for option in options:
        if option.get_attribute("value") == str(int(start_month_select.get_attribute("value")) - 1):
            select.select_by_visible_text(MONTHS[int(start_month_select.get_attribute("value")) - 1])
            search_button.click()
            break

time.sleep(5)

# Finding and activating the function to generate a table on the site
get_data_table_button = wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Genereaza Tabel')]")))
get_data_table_button.click()

# Wait for the table to load
table = wait.until(EC.presence_of_element_located((By.XPATH, "//table[@name='table_sen']")))

# # Wait for the XHR request to complete
# wait.until(EC.visibility_of_element_located((By.ID, "loading")))
# wait.until(EC.invisibility_of_element_located((By.ID, "loading")))

# # Get the XHR URL
# xhr_url = driver.find_element(By.XPATH, "//script[contains(text(), 'var table_sen')]").get_attribute("innerHTML").split("'")[-2]

# # Make a POST request to the XHR URL
# response = requests.post(xhr_url)

# # Parse the table data from the response
# table_data = []
# for row in response.json()['data']:
#     table_data.append(row)

# Extract the table data
table_rows = table.find_elements(By.TAG_NAME, "tr")
table_data = []
for row in table_rows:
    cols = row.find_elements(By.TAG_NAME, "td")
    cols = [col.text for col in cols]
    table_data.append(cols)

driver.quit()

# Convert the table data to a pandas DataFrame
df = pd.DataFrame(table_data)

# Save the DataFrame as a CSV file
df.to_csv("energy_data.csv", index=False)


