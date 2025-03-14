import pandas as pd

# Load Data
labels = pd.read_csv('prepared_instagram_data.csv')['Label']
features = pd.read_csv('tfidf_features.csv')  # Change this if your features are in a different file

# Check initial shapes
print(f"Initial Labels shape: {labels.shape}, Features shape: {features.shape}")

# Show unique values
print("🔍 Unique values in labels:\n", labels.unique())

# Find missing values
print("🔍 Missing values in labels before processing:\n", labels.isnull().sum())

# Drop NaNs
labels = labels.dropna()
print(f"After dropna - Labels shape: {labels.shape}")

# Map labels
labels = labels.replace({'Scam': 1, 'Not Scam': 0})
print("✅ Encoded Labels:\n", labels.value_counts())

# Show final shape
print(f"Final Labels shape: {labels.shape}, Features shape: {features.shape}")

print("🔍 Unique values in labels after encoding:\n", labels.value_counts(dropna=False))
if labels.isnull().sum() > 0:
    print("⚠️ Warning: Some labels were not mapped properly. Check label values.")
    print(labels[labels.isnull()])
