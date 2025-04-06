import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import streamlit as st
from datetime import datetime

# Load the cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")
    # Convert date column if exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
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
        default=sorted(df['bedrooms'].dropna().unique())
    )
    
    # Bathroom filter
    bathrooms = st.multiselect(
        'Number of Bathrooms',
        options=sorted(df['bathrooms'].dropna().unique()),
        default=sorted(df['bathrooms'].dropna().unique())
    )
    
    # Location filter (if available)
    if 'location' in df.columns:
        locations = st.multiselect(
            'Select Locations',
            options=df['location'].unique(),
            default=df['location'].unique()
        )
    
    # Property type filter (if available)
    if 'property_type' in df.columns:
        property_types = st.multiselect(
            'Property Type',
            options=df['property_type'].unique(),
            default=df['property_type'].unique()
        )
    
    # Year built filter (if available)
    if 'year_built' in df.columns:
        year_range = st.slider(
            'Year Built Range',
            min_value=int(df['year_built'].min()),
            max_value=int(df['year_built'].max()),
            value=(int(df['year_built'].min()), int(df['year_built'].max()))
        )
    
    # Date range filter (if available)
    if 'date' in df.columns:
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

# Filter dataframe based on selections
filtered_df = df[
    (df['price'] >= price_range[0]) & 
    (df['price'] <= price_range[1]) &
    (df['bedrooms'].isin(bedrooms)) &
    (df['bathrooms'].isin(bathrooms))
]

# Additional filters if columns exist
if 'location' in df.columns:
    filtered_df = filtered_df[filtered_df['location'].isin(locations)]
if 'property_type' in df.columns:
    filtered_df = filtered_df[filtered_df['property_type'].isin(property_types)]
if 'year_built' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['year_built'] >= year_range[0]) & 
        (filtered_df['year_built'] <= year_range[1])
    ]
if 'date' in df.columns and len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['date'].dt.date >= date_range[0]) & 
        (filtered_df['date'].dt.date <= date_range[1])
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

# 3. Enhanced Correlation Analysis -----------------------------------------------
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

# 4. Time Series Analysis (if date column exists) --------------------------------
if 'date' in df.columns:
    st.subheader("📅 Price Trends Over Time")
    
    # Resample by month or week
    time_resolution = st.radio(
        "Time Resolution",
        ["Monthly", "Weekly"],
        horizontal=True
    )
    
    if time_resolution == "Monthly":
        time_df = filtered_df.set_index('date').resample('M')['price'].mean().reset_index()
    else:
        time_df = filtered_df.set_index('date').resample('W')['price'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=time_df, x='date', y='price', color='crimson', ax=ax)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    plt.title(f'Average Price Over Time ({time_resolution})', pad=15, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Price ($)', fontsize=12)
    plt.grid(alpha=0.3)
    st.pyplot(fig)

# 5. Categorical Analysis -------------------------------------------------------
st.subheader("🏘️ Categorical Analysis")
cat_cols = [col for col in ['property_type', 'location', 'bedrooms', 'bathrooms'] if col in filtered_df.columns]

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

# 6. Price per Square Foot Analysis ---------------------------------------------
st.subheader("📐 Price per Square Foot Analysis")
if 'square_footage' in filtered_df.columns:
    filtered_df['price_per_sqft'] = filtered_df['price'] / filtered_df['square_footage']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(filtered_df['price_per_sqft'].dropna(), bins=30, kde=True, color='purple')
    plt.title('Distribution of Price per Square Foot', pad=15)
    plt.xlabel('Price per Square Foot ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.3)
    st.pyplot(fig)

# 7. Top/Bottom Properties ------------------------------------------------------
st.subheader("🏆 Top & Bottom Properties")
top_n = st.slider('Select number of properties to display', 1, 20, 5)

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Top {top_n} Most Expensive Properties**")
    st.dataframe(
        filtered_df.nlargest(top_n, 'price')[['price', 'square_footage', 'bedrooms', 'bathrooms'] + 
        (['location'] if 'location' in filtered_df.columns else []) + 
        (['property_type'] if 'property_type' in filtered_df.columns else [])]
        .style.format({'price': '${:,.0f}', 'square_footage': '{:,.0f}'})
    )

with col2:
    st.write(f"**Top {top_n} Best Value (Lowest Price per Sq Ft)**")
    if 'price_per_sqft' in filtered_df.columns:
        st.dataframe(
            filtered_df.nsmallest(top_n, 'price_per_sqft')[['price', 'square_footage', 'price_per_sqft', 'bedrooms', 'bathrooms'] + 
            (['location'] if 'location' in filtered_df.columns else []) + 
            (['property_type'] if 'property_type' in filtered_df.columns else [])]
            .style.format({'price': '${:,.0f}', 'square_footage': '{:,.0f}', 'price_per_sqft': '${:,.2f}'})
        )

# 8. Map Visualization (if coordinates available) -------------------------------
if all(col in filtered_df.columns for col in ['latitude', 'longitude']):
    st.subheader("🗺️ Property Map")
    st.map(filtered_df[['latitude', 'longitude', 'price']].dropna())
elif 'location' in filtered_df.columns:
    # Could add a choropleth map here if you have geospatial data
    pass

# 9. Pairwise Relationships (Bonus) --------------------------------------------
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

# Explicitly pass the figure to st.pyplot()
st.pyplot(pairplot_fig.fig)