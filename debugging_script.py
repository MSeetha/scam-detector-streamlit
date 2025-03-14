import pandas as pd
import numpy as np

# Load Features & Labels
features = pd.read_csv('tfidf_features.csv')
labels = pd.read_csv('prepared_instagram_data.csv')[['Label']]  # Ensure it's a DataFrame

# Debug Step 1: Check for NaN values
print("🔍 Missing values in labels before cleaning:\n", labels.isnull().sum())

# Drop NaN values in labels
labels.dropna(inplace=True)

# Debug Step 2: Ensure labels align with features
print(f"📊 Features shape: {features.shape}")
print(f"📊 Labels shape: {labels.shape}")

# Debug Step 3: Unique values in labels
print("🔍 Unique values in labels:\n", labels['Label'].unique())

# Debug Step 4: Check for empty labels
if labels.empty:
    print("⚠️ Labels DataFrame is empty after dropping NaNs!")

# Debug Step 5: Convert labels to numerical format
labels = labels.replace({'Scam': 1, 'Legit': 0})

# Debug Step 6: Ensure no NaN remains in labels
print("🔍 Missing values in labels after cleaning:\n", labels.isnull().sum())

# Debug Step 7: Convert labels to NumPy array and check data type
labels = labels.values.flatten()
print(f"✅ Labels converted to NumPy array. Data type: {labels.dtype}")
