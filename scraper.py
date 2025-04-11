from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from fake_useragent import UserAgent
import os

# Set up Selenium options
options = Options()
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--headless")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--window-size=1920,1080")
ua = UserAgent()
options.add_argument(f"user-agent={ua.random}")

# Start WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Base URL
base_url = "https://www.century21.com/real-estate/chicago-il/LCILCHICAGO/?beds=1&baths=1&minsqft=200"

# Lists to store data
all_data = []

# Loop through pages
for offset in range(0, 240, 24):
    url = base_url if offset == 0 else f"{base_url}&s={offset}"
    print(f"Scraping page with offset: {offset}")

    driver.get(url)
    time.sleep(2)  # Initial delay

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "property-card"))
        )

        listings = driver.find_elements(By.CLASS_NAME, "property-card")

        for listing in listings:
            data = {
                "Bedrooms": None,
                "Bathrooms": None,
                "Square Footage": None,
                "Address": None,
                "Price": None
            }

            try:
                data["Bedrooms"] = listing.find_element(By.CLASS_NAME, "property-spec.beds").text
            except:
                print("Bedrooms not found in listing")

            try:
                # Try multiple selectors for bathrooms
                try:
                    data["Bathrooms"] = listing.find_element(By.CSS_SELECTOR, ".specs-number.full-baths").text
                except:
                    data["Bathrooms"] = listing.find_element(By.CSS_SELECTOR, ".property-baths").text
            except:
                print("Bathrooms not found in listing")

            try:
                data["Square Footage"] = listing.find_element(By.CLASS_NAME, "property-spec.square-footage").text
            except:
                print("Square footage not found in listing")

            try:
                data["Address"] = listing.find_element(By.CLASS_NAME, "property-address").text.strip().replace("\n", " ")
            except:
                print("Address not found in listing")

            try:
                data["Price"] = listing.find_element(By.CLASS_NAME, "font-family-taglines.property-price").text
            except:
                print("Price not found in listing")

            all_data.append(data)

    except Exception as e:
        print(f"Error processing page with offset {offset}: {str(e)}")

    time.sleep(3)  # Delay between pages

# Create DataFrame
df = pd.DataFrame(all_data)

# Save raw data
df.to_csv("raw_data.csv", index=False)
print(f"Scraping complete. Saved {len(df)} listings to raw_data.csv")

driver.quit()