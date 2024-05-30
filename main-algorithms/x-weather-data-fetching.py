import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Download directory
DOWNLOAD_DIR = "/path/to/download"  # Change this to your desired download directory

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    prefs = {"download.default_directory": DOWNLOAD_DIR, "directory_upgrade": True}
    chrome_options.add_experimental_option("prefs", prefs)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def wait_for_element(driver, by, value, timeout=30):
    return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by, value)))

def switch_to_iframe_if_needed(driver, by, value):
    iframes = driver.find_elements(By.TAG_NAME, 'iframe')
    for iframe in iframes:
        driver.switch_to.frame(iframe)
        try:
            driver.find_element(by, value)
            return True
        except NoSuchElementException:
            driver.switch_to.default_content()
    return False

def download_weather_data(driver):
    url = 'https://power.larc.nasa.gov/data-access-viewer/'
    driver.get(url)
    logger.info("Opened the webpage.")

    # Allow the page to load completely
    time.sleep(5)

    # Switch to the iframe if necessary
    if not switch_to_iframe_if_needed(driver, By.ID, 'user_community'):
        logger.error("Unable to find the user_community element within the iframes.")
        return

    # Choose a User Community
    user_community_select = wait_for_element(driver, By.ID, 'user_community')
    Select(user_community_select).select_by_visible_text('Renewable Energy')
    logger.info("Selected User Community: Renewable Energy")

    # Choose a Temporal Average
    temporal_average_select = wait_for_element(driver, By.ID, 'temporal_average')
    Select(temporal_average_select).select_by_visible_text('Hourly')
    logger.info("Selected Temporal Average: Hourly")

    # Enter Latitude and Longitude
    lat_input = wait_for_element(driver, By.ID, 'lat')
    lat_input.clear()
    lat_input.send_keys('44.430')
    logger.info("Entered Latitude: 44.430")

    lon_input = wait_for_element(driver, By.ID, 'lon')
    lon_input.clear()
    lon_input.send_keys('26.113')
    logger.info("Entered Longitude: 26.113")

    # Enter Start Date
    start_date_input = wait_for_element(driver, By.ID, 'start_date')
    start_date_input.clear()
    start_date_input.send_keys('01/01/2022')
    logger.info("Entered Start Date: 01/01/2022")

    # Enter End Date
    end_date_input = wait_for_element(driver, By.ID, 'end_date')
    end_date_input.clear()
    end_date_input.send_keys('12/31/2022')
    logger.info("Entered End Date: 12/31/2022")

    # Select Output File Format
    output_format_select = wait_for_element(driver, By.ID, 'output_format')
    Select(output_format_select).select_by_visible_text('CSV')
    logger.info("Selected Output File Format: CSV")

    # Select Parameters (Assuming Solar Fluxes and Related -> Solar Irradiance)
    parameters_search = wait_for_element(driver, By.ID, 'parameters_search')
    parameters_search.send_keys('Solar Irradiance')
    logger.info("Searched for Parameter: Solar Irradiance")

    # Wait for search results to populate and select parameter
    time.sleep(2)
    solar_irradiance_checkbox = wait_for_element(driver, By.XPATH, "//label[text()='Solar Irradiance']/preceding-sibling::input")
    solar_irradiance_checkbox.click()
    logger.info("Selected Parameter: Solar Irradiance")

    # Submit the form
    submit_button = wait_for_element(driver, By.ID, 'submit_button')
    submit_button.click()
    logger.info("Submitted the form.")

    # Allow time for the file to be generated and downloaded
    time.sleep(60)
    logger.info("Waited for the file to be generated and downloaded.")

def main():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Set up the WebDriver
            driver = setup_driver()
            logger.info("WebDriver setup complete.")

            # Download the weather data
            download_weather_data(driver)
            logger.info("Weather data download process completed successfully.")
            break
        except (TimeoutException, WebDriverException) as e:
            logger.error(f"An error occurred (attempt {attempt + 1} of {max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                time.sleep(5)
            else:
                logger.error("Max retries reached. Exiting.")
        finally:
            driver.quit()
            logger.info("Closed the WebDriver.")

if __name__ == "__main__":
    # Ensure the download directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    main()
