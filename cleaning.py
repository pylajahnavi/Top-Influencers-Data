# # importing modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # Load the dataset

df = pd.read_csv('top_insta_influencers_data.csv')

# # Quick inspection of data

print(df.info())
print(df.describe())


# # Drop any duplicate rows if present

df.drop_duplicates(inplace=True)
df

# Handle missing values
# Fill missing numerical values with median, and categorical with mode
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is categorical
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:  # Assume the column is numerical
        df[column].fillna(df[column].median(), inplace=True)

# Clean and convert necessary columns to appropriate data types
def clean_and_convert(value):
    if isinstance(value, str):
        value = value.replace('m', '').replace('k', '').replace('b', '')
        if value == '':  # Handle empty strings
            return 0
        return float(value)
    return value

# Apply cleaning and conversion to relevant columns
df['followers'] = df['followers'].apply(clean_and_convert).astype(float)
df['posts'] = df['posts'].apply(clean_and_convert).astype(float)
df['total_likes'] = df['total_likes'].apply(clean_and_convert).astype(float)
df['avg_likes'] = df['avg_likes'].apply(clean_and_convert).astype(float)
df['new_post_avg_like'] = df['new_post_avg_like'].apply(clean_and_convert).astype(float)

# Convert to integers if needed
df['followers'] = df['followers'].astype(int)
df['posts'] = df['posts'].astype(int)
df['total_likes'] = df['total_likes'].astype(int)

# Step-2: Exploratory Data Analysis (EDA)
print(df[['influence_score', 'followers', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like']].describe())

# Relationship between Followers and Engagement
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='followers', y='60_day_eng_rate', hue='country', alpha=0.7)
plt.title('Followers vs 60-Day Engagement Rate')
plt.xlabel('Number of Followers')
plt.ylabel('60-Day Engagement Rate (%)')
plt.legend(title='Country')
plt.show()

# Distribution of Influence Score
plt.figure(figsize=(10, 5))
sns.histplot(df['influence_score'], bins=30, kde=True)
plt.title('Distribution of Influence Score')
plt.xlabel('Influence Score')
plt.show()

# Most active countries
top_countries = df['country'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_countries.index, y=top_countries.values, palette="viridis")
plt.title('Top 10 Countries by Number of Influencers')
plt.xlabel('Country')
plt.ylabel('Number of Influencers')
plt.show()

# Step-3: Feature Engineering
df['like_follower_ratio'] = df['total_likes'] / df['followers']
df['post_follower_ratio'] = df['posts'] / df['followers']
df['avg_likes_ratio'] = df['avg_likes'] / df['followers']

# Step-4 Model Building

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define feature columns and target variable
X = df[['followers', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like', 'like_follower_ratio', 'post_follower_ratio']]
y = df['influence_score']

# Clean and convert the '60_day_eng_rate' column
X['60_day_eng_rate'] = X['60_day_eng_rate'].str.replace('%', '').astype(float) / 100

# Check for missing values and handle them
print("Missing values in X before handling:")
print(X.isnull().sum())

# Fill missing values with the median
X.fillna(X.median(), inplace=True)

# Check for infinite values and handle them
print("Infinite values in X before handling:")
print(np.isinf(X).sum())

# Replace infinite values with NaN and then fill with the median
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Ensure all columns are numeric
print("Data types of X:")
print(X.dtypes)

# Convert any non-numeric columns to numeric (if needed)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill any new NaN values created during conversion
X.fillna(X.median(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()

# Scale the training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Model Interpretation and Feature Importance
# Display feature Importances

feature_importances = pd.Series(model.feature_importances_, index= X.columns)
feature_importances.sort_values().plot(kind= 'barh', title= 'Feature Importance')
plt.show()

#Visualizing Predictions

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha = 0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color = 'red')
plt.xlabel('True Infuence Score')
plt.ylabel('Predicted Influence Score')
plt.title('True vs Predicted Influence Score')
plt.show()
