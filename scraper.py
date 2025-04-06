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

# Set up Selenium options
options = Options()
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--headless")  # Uncomment for headless mode
options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid bot detection
options.add_argument("--window-size=1920,1080")  # Set a realistic window size
ua = UserAgent()
options.add_argument(f"user-agent={ua.random}")

# Start WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Base URL
base_url = "https://www.century21.com/real-estate/chicago-il/LCILCHICAGO/?beds=1&baths=1&minsqft=200"

# Lists to store raw data
all_beds, all_baths, all_sqft, all_adrs, all_prices = [], [], [], [], []

# Loop through the first 10 pages (from page 1 to 10)
for offset in range(0, 240, 24):
    url = base_url if offset == 0 else f"{base_url}&s={offset}"
    print(f"Scraping page with offset: {offset}")

    driver.get(url)

    try:
        # Wait for listings to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "property-card"))
        )

        # Find all listing elements
        listings = driver.find_elements(By.CLASS_NAME, "property-card")

        for listing in listings:
            try:
                # Extract raw text without cleaning
                beds = listing.find_element(By.CLASS_NAME, "property-spec.beds").text
                baths = listing.find_element(By.CLASS_NAME, "specs-number.full-baths").text
                sqft = listing.find_element(By.CLASS_NAME, "property-spec.square-footage").text
                address = listing.find_element(By.CLASS_NAME, "property-address").text.strip().replace("\n", " ")
                price = listing.find_element(By.CLASS_NAME, "font-family-taglines.property-price").text

                # Append raw text data (no formatting or conversion)
                all_beds.append(beds)
                all_baths.append(baths)
                all_sqft.append(sqft)
                all_adrs.append(address)
                all_prices.append(price)

            except Exception as e:
                print(f"Skipping a listing due to error: {e}")

    except Exception as e:
        print(f"Skipping page with offset {offset} due to error: {e}")

    time.sleep(2)  # Short delay to avoid bot detection

# Create a DataFrame with RAW data (no cleaning)
df = pd.DataFrame({
    "Bedrooms": all_beds,
    "Bathrooms": all_baths,
    "Square Footage": all_sqft,
    "Address": all_adrs,
    "Price": all_prices
})

# Save raw data to CSV (no modifications)
df.to_csv("webscraping_project_real_estate/raw_data.csv", index=False)
print("Scraping complete. **Raw data** saved to raw_data.csv.")

driver.quit()
