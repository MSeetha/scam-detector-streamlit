import pandas as pd

# Load the dataset
df_instagram = pd.read_csv("instagram_dataset.csv")

# Print column names to check
print("Columns in dataset:", df_instagram.columns)

# Remove missing values
df_instagram = df_instagram.dropna()

# Convert text in the 'Captions' column to lowercase
df_instagram['Captions'] = df_instagram['Captions'].astype(str).str.lower()

# Remove duplicate rows
df_instagram = df_instagram.drop_duplicates()

# Save cleaned dataset
df_instagram.to_csv("cleaned_instagram_data.csv", index=False)

print("Data cleaning completed. Saved as cleaned_instagram_data.csv")

