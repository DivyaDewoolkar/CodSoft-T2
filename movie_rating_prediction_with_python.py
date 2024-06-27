# -*- coding: utf-8 -*-
"""MOVIE RATING PREDICTION WITH PYTHON"""

# Step 1: Data Analysis and Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
try:
    movies_df = pd.read_csv('IMDb_Movies.csv', encoding='utf-8')
except UnicodeDecodeError:
    movies_df = pd.read_csv('IMDb_Movies.csv', encoding='latin1')

# Display the first few rows of the dataframe
print(movies_df.head())

# Display basic information about the dataframe
print(movies_df.info())

# Handle missing values (e.g., dropping or filling them)
movies_df = movies_df.dropna()

# Step 2: Feature Engineering

# Encode categorical variables
label_encoder = LabelEncoder()
movies_df['Genre'] = label_encoder.fit_transform(movies_df['Genre'])
movies_df['Director'] = label_encoder.fit_transform(movies_df['Director'])
# Check for the correct column names related to actors
print(movies_df.columns) # Print the columns to see the available options
# Assuming the actors are in columns 'Actor 1', 'Actor 2', 'Actor 3'
movies_df['Actor 1'] = label_encoder.fit_transform(movies_df['Actor 1'])
movies_df['Actor 2'] = label_encoder.fit_transform(movies_df['Actor 2'])
movies_df['Actor 3'] = label_encoder.fit_transform(movies_df['Actor 3'])

# Select features and target variable
# Adjust the features list to include the correct actor columns
features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
target = 'Rating'
X = movies_df[features]
y = movies_df[target]

# Step 3: Modeling

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Prediction

# Predict ratings on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.show()
