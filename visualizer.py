import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import re
from collections import Counter
from wordcloud import WordCloud


# Load the cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")
    return df

df = load_data()

# Ensure numeric columns are properly converted
numeric_cols = ['price', 'square_footage', 'bedrooms', 'bathrooms']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Set display options
pd.set_option('display.float_format', '{:,.2f}'.format)

# Title for Streamlit App
st.title("🏠 Advanced Real Estate Analytics Dashboard")

# Sidebar for Input Parameters
with st.sidebar:
    st.header("🔍 Filter Data")
    
    # Price range selection
    price_range = st.slider(
        'Select Price Range ($)',
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].min()), int(df['price'].max()))
    )
    
    # Bedroom filter
    bedrooms = st.multiselect(
        'Number of Bedrooms',
        options=sorted(df['bedrooms'].dropna().unique()),
        default=sorted(df['bedrooms'].dropna().unique()))
    
    # Bathroom filter
    bathrooms = st.multiselect(
        'Number of Bathrooms',
        options=sorted(df['bathrooms'].dropna().unique()),
        default=sorted(df['bathrooms'].dropna().unique()))

# Filter dataframe based on selections
filtered_df = df[
    (df['price'] >= price_range[0]) & 
    (df['price'] <= price_range[1]) &
    (df['bedrooms'].isin(bedrooms)) &
    (df['bathrooms'].isin(bathrooms))
]

# Display summary statistics
st.subheader("📊 Summary Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Properties", len(filtered_df))
col2.metric("Average Price", f"${filtered_df['price'].mean():,.0f}")
col3.metric("Average Size", f"{filtered_df['square_footage'].mean():,.0f} sq ft")
col4.metric("Price per Sq Ft", f"${(filtered_df['price']/filtered_df['square_footage']).mean():.2f}")

# Display DataFrame with expander
with st.expander("🔎 View Filtered Data"):
    st.dataframe(filtered_df)

# 1. Price Distribution Analysis --------------------------------------------------
st.subheader("💰 Price Distribution Analysis")
tab1, tab2 = st.tabs(["Histogram", "Boxplot"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(filtered_df['price'].dropna(), kde=True, bins=30, color='royalblue', ax=ax)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    plt.title('Price Distribution of Property Listings', pad=20, fontsize=14)
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    sns.despine()
    plt.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=filtered_df['price'].dropna(), color='lightseagreen', width=0.4, ax=ax)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    plt.title('Price Distribution with Outliers', pad=15, fontsize=14)
    plt.xlabel('Price ($)', fontsize=12)
    sns.despine()
    plt.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    
    # Calculate outlier thresholds
    Q1 = filtered_df['price'].quantile(0.25)
    Q3 = filtered_df['price'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    st.write(f"**Outlier threshold:** ${outlier_threshold:,.2f}")
    st.write(f"**Number of outliers:** {(filtered_df['price'] > outlier_threshold).sum()}")

# 2. Price vs Square Footage -----------------------------------------------------
st.subheader("📏 Price vs. Square Footage")
fig, ax = plt.subplots(figsize=(12, 7))
scatter = sns.scatterplot(
    data=filtered_df.dropna(),
    x='square_footage',
    y='price',
    hue='bedrooms',
    palette='viridis',
    size='bathrooms',
    sizes=(20, 200),
    alpha=0.7,
    ax=ax
)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
plt.title('Price vs. Square Footage (Colored by Bedrooms, Sized by Bathrooms)', pad=15, fontsize=14)
plt.xlabel('Square Footage (sq ft)', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.2)
st.pyplot(fig)

# 3.Correlation Analysis -----------------------------------------------
st.subheader("🔗 Feature Correlation Matrix")
corr_matrix = filtered_df[numeric_cols].corr()

# Plot the correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Custom diverging colormap
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap=cmap,
    center=0,
    fmt=".2f",
    linewidths=0.5,
    linecolor='white',
    annot_kws={"size": 11, "weight": "bold"},
    cbar_kws={"shrink": 0.75, "label": "Correlation Coefficient"},
    ax=ax
)

plt.title('Feature Correlation Matrix\n', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
st.pyplot(fig)

# 4. Categorical Analysis -------------------------------------------------------
st.subheader("🏘️ Categorical Analysis")
cat_cols = ['bedrooms', 'bathrooms']

if cat_cols:
    cat_choice = st.selectbox("Select category to analyze", cat_cols)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of average price by category
    sns.barplot(
        data=filtered_df,
        x=cat_choice,
        y='price',
        estimator=np.mean,
        ci=None,
        palette='coolwarm',
        ax=ax1
    )
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    ax1.set_title(f'Average Price by {cat_choice.title()}', pad=15)
    ax1.set_xlabel('')
    ax1.set_ylabel('Average Price ($)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Count plot of listings by category
    sns.countplot(
        data=filtered_df,
        x=cat_choice,
        palette='coolwarm',
        ax=ax2
    )
    ax2.set_title(f'Number of Listings by {cat_choice.title()}', pad=15)
    ax2.set_xlabel('')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

# 5. Price per Square Foot Analysis ---------------------------------------------
st.subheader("📐 Price per Square Foot Analysis")
filtered_df['price_per_sqft'] = filtered_df['price'] / filtered_df['square_footage']
    
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(filtered_df['price_per_sqft'].dropna(), bins=30, kde=True, color='purple')
plt.title('Distribution of Price per Square Foot', pad=15)
plt.xlabel('Price per Square Foot ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(alpha=0.3)
st.pyplot(fig)

# 6. Top/Bottom Properties ------------------------------------------------------
st.subheader("🏆 Top & Bottom Properties")
top_n = st.slider('Select number of properties to display', 1, 20, 5)

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Top {top_n} Most Expensive Properties**")
    st.dataframe(
        filtered_df.nlargest(top_n, 'price')[['price', 'square_footage', 'bedrooms', 'bathrooms']]
        .style.format({'price': '${:,.0f}', 'square_footage': '{:,.0f}'})
    )

with col2:
    st.write(f"**Top {top_n} Best Value (Lowest Price per Sq Ft)**")
    st.dataframe(
        filtered_df.nsmallest(top_n, 'price_per_sqft')[['price', 'square_footage', 'price_per_sqft', 'bedrooms', 'bathrooms']]
        .style.format({'price': '${:,.0f}', 'square_footage': '{:,.0f}', 'price_per_sqft': '${:,.2f}'})
    )

# 7. Pairwise Relationships--------------------------------------------
st.subheader("📈 Pairwise Feature Relationships")

# Create the pairplot figure explicitly
pairplot_fig = sns.pairplot(
    filtered_df[numeric_cols].dropna(),
    corner=True,
    plot_kws={'alpha': 0.6, 's': 15},
    diag_kws={'color': 'lightseagreen'},
)

# Add title and adjust layout
pairplot_fig.fig.suptitle('Pairwise Feature Relationships', y=1.02)
pairplot_fig.fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Render the pairplot
st.pyplot(pairplot_fig)

# 8. Machine Learning Price Prediction ----------------------------------------
st.subheader("🤖 Enhanced Machine Learning Price Prediction")

if len(filtered_df) > 100:
    # Feature selection
    ml_features = ['square_footage', 'bedrooms', 'bathrooms']
    
    # Prepare data
    ml_df = filtered_df[ml_features + ['price']].dropna()
    
    if len(ml_df) > 100:
        # Split data
        X = ml_df.drop('price', axis=1)
        y = ml_df['price']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model configuration
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Add progress bar
        with st.spinner('Training model...'):
            model.fit(X_train_scaled, y_train)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error", f"${mae:,.0f}", 
                   help="Average absolute difference between predicted and actual prices")
        col2.metric("R² Score", f"{r2:.3f}", 
                   help="Proportion of variance explained by the model (1 is perfect)")
        col3.metric("Mean Absolute % Error", f"{mape:.1f}%", 
                   help="Average percentage difference between predicted and actual prices")
        
        # Actual vs Predicted plot
        st.write("**Actual vs. Predicted Prices**")
        fig, ax = plt.subplots(figsize=(8, 6))
        max_price = max(y_test.max(), y_pred.max())
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
        ax.plot([0, max_price], [0, max_price], '--r')
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title('Actual vs. Predicted Prices')
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
        st.pyplot(fig)
        
        # Feature importance
        st.write("**Feature Importance**")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
        plt.title('Feature Importance in Price Prediction')
        st.pyplot(fig)
        
        # Prediction interface
        with st.expander("🔮 Make a Custom Prediction", expanded=True):
            st.write("Enter property details to estimate value:")
            
            col1, col2, col3 = st.columns(3)
            sqft = col1.number_input("Square Footage", min_value=100, max_value=10000, value=1500)
            beds = col2.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            baths = col3.number_input("Bathrooms", min_value=1, max_value=10, value=2)
            
            input_data = {
                'square_footage': sqft,
                'bedrooms': beds,
                'bathrooms': baths
            }
            
            if st.button("Estimate Price", type="primary"):
                # Prepare input in correct order
                input_df = pd.DataFrame([input_data])
                
                # Ensure columns match training data (fill missing with 0)
                for col in X.columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[X.columns]
                
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                
                st.success(f"Estimated Property Value: ${prediction:,.0f}")
                
                # Show comparable properties
                st.write("**Similar Properties in Dataset:**")
                comparable = filtered_df[
                    (filtered_df['square_footage'].between(sqft*0.9, sqft*1.1)) &
                    (filtered_df['bedrooms'] == beds) &
                    (filtered_df['bathrooms'] == baths)
                ]
                
                if not comparable.empty:
                    st.dataframe(
                        comparable.nsmallest(5, 'price')[['price', 'square_footage', 'bedrooms', 'bathrooms']]
                        .style.format({'price': '${:,.0f}', 'square_footage': '{:,.0f}'})
                    )
                else:
                    st.warning("No similar properties found in the dataset")
    else:
        st.warning(f"Not enough data for machine learning (need at least 100 valid samples, only {len(ml_df)} available)")
else:
    st.warning("Not enough data for machine learning (need at least 100 samples)")
    
# =============================================
# NLP FEATURES
# =============================================

# Function to extract zip codes from addresses
def extract_zip_code(address):
    zip_code = re.findall(r'IL (\d{5})', str(address))
    return zip_code[0] if zip_code else None

# function to extract street types
def extract_street_type(address):
    street_types = ['Street', 'Avenue', 'Road', 'Drive', 'Court', 'Place', 
                   'Boulevard', 'Way', 'Lane', 'Terrace', 'Circle', 'Highway',
                   'Parkway', 'Square', 'Trail', 'Alley', 'Plaza']
    
    # Look for street types in the address (case insensitive)
    address_str = str(address).title()  # Standardize capitalization
    for stype in street_types:
        if f" {stype}" in address_str:
            return stype
    
    # Check abbreviated forms
    abbrev_map = {
        'St': 'Street',
        'Ave': 'Avenue',
        'Rd': 'Road',
        'Dr': 'Drive',
        'Ct': 'Court',
        'Pl': 'Place',
        'Blvd': 'Boulevard',
        'Ln': 'Lane',
        'Ter': 'Terrace',
        'Cir': 'Circle',
        'Hwy': 'Highway',
        'Pkwy': 'Parkway',
        'Sq': 'Square',
        'Trl': 'Trail',
        'Aly': 'Alley'
    }
    
    for abbrev, full in abbrev_map.items():
        if f" {abbrev}" in address_str or f" {abbrev}." in address_str:
            return full
    
    return 'Other'

# Function to extract neighborhood/area
def extract_area(address):
    # Common Chicago neighborhoods and their patterns
    neighborhood_patterns = {
        'Loop': r'\d+ (N|S|E|W) \w+ (St|Street|Ave|Avenue|Dr|Drive|Blvd|Boulevard)',
        'Gold Coast': r'\d+ N (Lake|State|Dearborn|Clark|Rush|Astor)',
        'Lincoln Park': r'\d+ N (Clark|Halsted|Sheffield|Lincoln|Damen|Racine)',
        'Wicker Park': r'\d+ N (Milwaukee|Damen|North|Wood|Ashland)',
        'Logan Square': r'\d+ N (Kedzie|Milwaukee|California|Sacramento)',
        'River North': r'\d+ N (Franklin|Orleans|Wells|LaSalle|Hubbard)',
        'South Loop': r'\d+ S (State|Wabash|Michigan|Dearborn|Clark)',
        'West Loop': r'\d+ W (Madison|Washington|Randolph|Lake|Fulton)',
        'Hyde Park': r'\d+ (E|W) \d+ (St|Street)',
        'Pilsen': r'\d+ W \d+ (St|Street)',
        'Edgewater': r'\d+ N (Broadway|Sheridan|Winthrop|Kenmore)',
        'Lakeview': r'\d+ N (Broadway|Halsted|Sheffield|Ashland|Lincoln)',
        'Uptown': r'\d+ N (Broadway|Clark|Sheridan|Winthrop)',
        'Andersonville': r'\d+ N (Clark|Ashland|Broadway)',
        'Bucktown': r'\d+ N (Damen|Western|Milwaukee)',
        'Wrigleyville': r'\d+ N (Clark|Sheffield|Addison|Newport)'
    }
    
    address_str = str(address)
    for neighborhood, pattern in neighborhood_patterns.items():
        if re.search(pattern, address_str, re.IGNORECASE):
            return neighborhood
    
    # Fallback to directional area if no specific neighborhood found
    if re.search(r'\d+ N \w+', address_str):
        return 'North Side'
    elif re.search(r'\d+ S \w+', address_str):
        return 'South Side'
    elif re.search(r'\d+ W \w+', address_str):
        return 'West Side'
    elif re.search(r'\d+ E \w+', address_str):
        return 'East Side'
    
    return 'Other Area'

# Apply NLP functions to create new features
filtered_df['zip_code'] = filtered_df['address'].apply(extract_zip_code)
filtered_df['street_type'] = filtered_df['address'].apply(extract_street_type)
filtered_df['area'] = filtered_df['address'].apply(extract_area)

# =============================================
# NLP VISUALIZATIONS
# =============================================

# 1. Location Analysis by Area
st.subheader("🗺️ Location Analysis by Area")

if len(filtered_df['area'].unique()) > 1:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Average price by area
    area_price = filtered_df.groupby('area')['price'].mean().sort_values(ascending=False)
    sns.barplot(x=area_price.values, y=area_price.index, palette='viridis', ax=ax1)
    ax1.set_title('Average Price by Neighborhood/Area', fontsize=14)
    ax1.set_xlabel('Average Price ($)', fontsize=12)
    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    ax1.tick_params(axis='y', labelsize=10)
    
    # Price distribution by area
    sns.boxplot(
        data=filtered_df,
        x='price',
        y='area',
        order=area_price.index,
        palette='viridis',
        ax=ax2
    )
    ax2.set_title('Price Distribution by Neighborhood/Area', fontsize=14)
    ax2.set_xlabel('Price ($)', fontsize=12)
    ax2.set_ylabel('')
    ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    ax2.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show top areas by price per sqft
    filtered_df['price_per_sqft'] = filtered_df['price'] / filtered_df['square_footage']
    area_pps = filtered_df.groupby('area')['price_per_sqft'].mean().sort_values(ascending=False).head(10)
    
    st.write("**Top Areas by Price per Square Foot:**")
    st.dataframe(
        area_pps.reset_index().rename(columns={'price_per_sqft': 'Price per SqFt ($)'})
        .style.format({'Price per SqFt ($)': '${:,.2f}'})
    )
else:
    st.warning("Not enough area diversity to display location analysis")

# 2. Zip Code Analysis
st.subheader("📮 Zip Code Analysis")

if len(filtered_df['zip_code'].unique()) > 1:
    # Get top 10 zip codes by count
    top_zip_codes = filtered_df['zip_code'].value_counts().nlargest(10).index
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Average price by zip code (top 10)
    zip_price = filtered_df[filtered_df['zip_code'].isin(top_zip_codes)].groupby('zip_code')['price'].mean().sort_values(ascending=False)
    sns.barplot(x=zip_price.values, y=zip_price.index, palette='coolwarm', ax=ax1)
    ax1.set_title('Average Price by Top 10 Zip Codes', fontsize=14)
    ax1.set_xlabel('Average Price ($)', fontsize=12)
    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    
    # Price distribution by zip code
    sns.boxplot(
        data=filtered_df[filtered_df['zip_code'].isin(top_zip_codes)],
        x='price',
        y='zip_code',
        order=zip_price.index,
        palette='coolwarm',
        ax=ax2
    )
    ax2.set_title('Price Distribution by Top 10 Zip Codes', fontsize=14)
    ax2.set_xlabel('Price ($)', fontsize=12)
    ax2.set_ylabel('Zip Code', fontsize=12)
    ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show zip code summary statistics
    st.write("**Zip Code Statistics:**")
    zip_stats = filtered_df.groupby('zip_code').agg({
        'price': ['count', 'mean', 'median', 'min', 'max'],
        'square_footage': 'mean',
        'price_per_sqft': 'mean'
    }).sort_values(('price', 'mean'), ascending=False)
    
    st.dataframe(
        zip_stats.style.format({
            ('price', 'mean'): '${:,.0f}',
            ('price', 'median'): '${:,.0f}',
            ('price', 'min'): '${:,.0f}',
            ('price', 'max'): '${:,.0f}',
            ('square_footage', 'mean'): '{:,.0f}',
            ('price_per_sqft', 'mean'): '${:,.2f}'
        })
    )
else:
    st.warning("Not enough zip code diversity to display analysis")

# 3. Street Type Analysis
st.subheader("🛣️ Street Type Analysis")

if len(filtered_df['street_type'].unique()) > 1:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # Average price by street type
    street_price = filtered_df.groupby('street_type')['price'].mean().sort_values(ascending=False)
    sns.barplot(x=street_price.values, y=street_price.index, palette='magma', ax=ax1)
    ax1.set_title('Average Price by Street Type', fontsize=14)
    ax1.set_xlabel('Average Price ($)', fontsize=12)
    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    
    # Count of listings by street type
    street_count = filtered_df['street_type'].value_counts()
    sns.barplot(x=street_count.values, y=street_count.index, palette='magma', ax=ax2)
    ax2.set_title('Number of Listings by Street Type', fontsize=14)
    ax2.set_xlabel('Count', fontsize=12)
    
    # Price per sqft by street type
    street_pps = filtered_df.groupby('street_type')['price_per_sqft'].mean().sort_values(ascending=False)
    sns.barplot(x=street_pps.values, y=street_pps.index, palette='magma', ax=ax3)
    ax3.set_title('Price per SqFt by Street Type', fontsize=14)
    ax3.set_xlabel('Price per SqFt ($)', fontsize=12)
    ax3.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.2f}'))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show street type statistics
    st.write("**Street Type Statistics:**")
    st.dataframe(
        filtered_df.groupby('street_type').agg({
            'price': ['mean', 'median', 'count'],
            'square_footage': 'mean',
            'price_per_sqft': 'mean'
        }).sort_values(('price', 'mean'), ascending=False)
        .style.format({
            ('price', 'mean'): '${:,.0f}',
            ('price', 'median'): '${:,.0f}',
            ('square_footage', 'mean'): '{:,.0f}',
            ('price_per_sqft', 'mean'): '${:,.2f}'
        })
    )
else:
    st.warning("Not enough street type diversity to display analysis")

# 4. Word Cloud of Street Names
st.subheader("☁️ Street Name Word Cloud")

# Extract street names (excluding street types)
def extract_street_names(address):
    address_str = str(address)
    # Remove unit numbers, city, state, zip
    address_str = re.sub(r'(\d+[A-Za-z]*,)|(,.*IL \d{5})', '', address_str)
    # Remove street types
    for stype in ['Street', 'Avenue', 'Road', 'Drive', 'Court', 'Place', 
                 'Boulevard', 'Way', 'Lane', 'Terrace']:
        address_str = address_str.replace(stype, '')
    return address_str.strip()

street_names = filtered_df['address'].apply(extract_street_names).str.cat(sep=' ')

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white',
                      colormap='viridis',
                      stopwords=['North', 'South', 'East', 'West', 'N', 'S', 'E', 'W']).generate(street_names)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Most Common Street Names in Property Addresses', fontsize=14, pad=20)
st.pyplot(fig)

# 5. Price vs. Street Type and Area
st.subheader("📊 Price by Street Type and Area")

if len(filtered_df['street_type'].unique()) > 1 and len(filtered_df['area'].unique()) > 1:
    # Get top 5 street types and areas by count
    top_street_types = filtered_df['street_type'].value_counts().nlargest(5).index
    top_areas = filtered_df['area'].value_counts().nlargest(5).index
    
    filtered_combo = filtered_df[
        filtered_df['street_type'].isin(top_street_types) & 
        filtered_df['area'].isin(top_areas)
    ]
    
    if not filtered_combo.empty:
        # Create figure with larger size
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Use pointplot instead of boxplot for better readability with many categories
        sns.pointplot(
            data=filtered_combo,
            x='street_type',
            y='price',
            hue='area',
            palette='tab10',
            estimator=np.median,
            errorbar=('ci', 95),  # Show 95% confidence intervals
            markers='o',
            linestyles='-',
            dodge=0.4,  # Separate points for each area
            scale=1.2,  # Increase point size
            ax=ax
        )
        
        # Formatting
        ax.set_title('Median Price by Street Type and Area\n(with 95% Confidence Intervals)', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Street Type', fontsize=14)
        ax.set_ylabel('Price ($)', fontsize=14)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # legend
        ax.legend(title='Neighborhood Area', 
                 bbox_to_anchor=(1.05, 1), 
                 loc='upper left',
                 fontsize=12,
                 title_fontsize=13)
        
        # gridlines for better value reading
        ax.grid(True, axis='y', alpha=0.3)
        
        # Remove spines for cleaner look
        sns.despine()
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        st.pyplot(fig)
        
        # explanatory text
        st.markdown("""
        **How to interpret this chart:**
        - Each point represents the **median price** for properties on that street type in that area
        - The lines show the 95% confidence interval around the median estimate
        - Compare how the same street type varies across different areas
        - Look for street types that consistently command higher prices across areas
        """)
        
        # sample size information
        st.write(f"**Sample sizes:** Total {len(filtered_combo)} properties ({len(top_street_types)} street types × {len(top_areas)} areas)")
        
    else:
        st.warning("Not enough data to show combination of street type and area")
else:
    st.warning("Need both street type and area diversity to display this analysis")