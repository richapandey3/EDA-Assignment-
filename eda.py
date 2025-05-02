# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set(style="whitegrid")

# --- 1. Used Bikes Dataset ---
print("\n--- Used Bikes Dataset ---")
bike_df = pd.read_csv("BIKE DETAILS (1).csv")

print(bike_df.info())
print(bike_df.describe())
print(bike_df['selling_price'].describe())

# Median selling price
print("Median selling price:", bike_df['selling_price'].median())

# Most common seller type
print("Most common seller type:", bike_df['seller_type'].mode()[0])

# Bikes driven more than 50,000 km
print("Bikes > 50,000 km:", (bike_df['km_driven'] > 50000).sum())

# Average km_driven by owner type
print("Average km_driven per owner type:\n", bike_df.groupby('owner')['km_driven'].mean())

# Proportion of bikes from 2015 or older
print("Proportion from 2015 or earlier:", (bike_df['year'] <= 2015).mean())

# Max showroom price and related bike
max_price = bike_df['ex_showroom_price'].max()
print("Highest ex_showroom_price:", max_price)
print("Bike with max ex_showroom_price:\n", bike_df[bike_df['ex_showroom_price'] == max_price][['name', 'ex_showroom_price']])

# Seller type counts
print("Bikes listed per seller type:\n", bike_df['seller_type'].value_counts())

# Selling price vs km_driven for 1st owner
sns.scatterplot(data=bike_df[bike_df['owner'] == '1st owner'], x='km_driven', y='selling_price')
plt.title("Selling Price vs KM Driven for 1st Owner Bikes")
plt.show()

# Remove outliers in km_driven
Q1 = bike_df['km_driven'].quantile(0.25)
Q3 = bike_df['km_driven'].quantile(0.75)
IQR = Q3 - Q1
bike_df_clean = bike_df[(bike_df['km_driven'] >= Q1 - 1.5 * IQR) & (bike_df['km_driven'] <= Q3 + 1.5 * IQR)]

# Bivariate plot: year vs selling_price
sns.boxplot(data=bike_df_clean, x='year', y='selling_price')
plt.xticks(rotation=45)
plt.title("Selling Price by Year")
plt.show()

# Depreciation = ex_showroom_price - selling_price by age
bike_df['age'] = 2025 - bike_df['year']
bike_df['depreciation'] = bike_df['ex_showroom_price'] - bike_df['selling_price']
print("Avg Depreciation by Age:\n", bike_df.groupby('age')['depreciation'].mean())

# Correlation heatmap
sns.heatmap(bike_df[['selling_price', 'km_driven', 'ex_showroom_price', 'age']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix (Used Bikes)")
plt.show()


# --- 2. Amazon Sales Dataset ---
print("\n--- Amazon Dataset ---")
amazon_df = pd.read_csv("amazon.csv")
print(amazon_df.info())
print(amazon_df.describe())

# Average rating per category
print("Average rating per category:\n", amazon_df.groupby("category")["rating"].mean())

# Top products by rating count per category
top_products = amazon_df.sort_values(by="rating_count", ascending=False).groupby("category").head(1)
print("Top products by rating_count:\n", top_products[['product_name', 'category', 'rating_count']])

# Price distribution plot
sns.histplot(data=amazon_df, x="discounted_price", color="green", label="Discounted", kde=True)
sns.histplot(data=amazon_df, x="actual_price", color="red", label="Actual", kde=True)
plt.legend()
plt.title("Discounted vs Actual Price")
plt.show()

# Correlation between discount and rating
print("Correlation (discounted price vs rating):", amazon_df[['discounted_price', 'rating']].corr())


# --- 3. Spotify Dataset ---
print("\n--- Spotify Dataset ---")
spotify_df = pd.read_csv("spotify.csv")
spotify_df.drop_duplicates(inplace=True)
spotify_df.dropna(inplace=True)

# Popularity distribution
sns.histplot(data=spotify_df, x='Popularity', bins=20, kde=True)
plt.title("Distribution of Track Popularity")
plt.show()

# Popularity vs duration
sns.scatterplot(data=spotify_df, x='Duration (ms)', y='Popularity')
plt.title("Popularity vs Duration")
plt.show()

# Most frequent artists
plt.figure(figsize=(10,5))
sns.countplot(data=spotify_df, y='Artist', order=spotify_df['Artist'].value_counts().head(10).index)
plt.title("Top 10 Artists by Track Count")
plt.show()

# Least popular tracks
print("Least Popular Tracks:\n", spotify_df.sort_values(by='Popularity').head(5)[['Artist', 'Track Name']])

# Top 5 artists with highest avg popularity
top_artists = spotify_df.groupby('Artist')['Popularity'].mean().sort_values(ascending=False).head(5)
print("Top 5 Artists by Avg Popularity:\n", top_artists)

# Popular tracks for top artists
for artist in top_artists.index:
    top_track = spotify_df[spotify_df['Artist'] == artist].sort_values(by='Popularity', ascending=False).head(1)
    print(f"{artist}: {top_track['Track Name'].values[0]}")

# Pair plot
sns.pairplot(spotify_df[['Popularity', 'Duration (ms)']])
plt.show()

# Box plot for duration by artist
top_artist_names = top_artists.index.tolist()
sns.boxplot(data=spotify_df[spotify_df['Artist'].isin(top_artist_names)], x='Artist', y='Duration (ms)')
plt.title("Track Duration by Top Artists")
plt.show()


# --- 4. Used Cars Dataset ---
print("\n--- Used Cars Dataset ---")
car_df = pd.read_csv("Car Sale.csv")
print(car_df.info())
print(car_df.describe())

# You can apply similar steps for car_df as done above
