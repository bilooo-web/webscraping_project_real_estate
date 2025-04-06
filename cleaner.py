import pandas as pd
import re

# Load raw data
df = pd.read_csv("webscraping_project_real_estate/raw_data.csv")

# Normalize column names to match `scraper.py`
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Function to extract numeric values from text
def extract_numeric(value):
    if pd.isna(value):
        return None
    match = re.search(r'[\d,]+', str(value))  # Extract first numeric pattern
    return int(match.group(0).replace(',', '')) if match else None

# Clean relevant columns
for col in ['price', 'bedrooms', 'bathrooms', 'square_footage']:
    df[col] = df[col].astype(str).apply(extract_numeric)

# Drop listings without valid prices
df.dropna(subset=['price'], inplace=True)

# Remove outliers using IQR method for price
Q1, Q3 = df['price'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save cleaned data
df.to_csv("webscraping_project_real_estate/cleaned_data.csv", index=False)
print("Data cleaning complete. Cleaned data saved to cleaned_data.csv.")
