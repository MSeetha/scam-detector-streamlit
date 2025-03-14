import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
file_path = "prepared_instagram_data.csv"
df = pd.read_csv(file_path)

# Display original class distribution
print("✅ Original class distribution:\n", df["Label"].value_counts())

# Separate features and labels
X = df["Captions"]  # Text data
y = df["Label"]  # Target labels

# Save raw balanced dataset (before vectorization)
df.to_csv("balanced_instagram_data_raw.csv", index=False)
print("\n✅ Balanced dataset (raw text) saved as 'balanced_instagram_data_raw.csv'")

# Convert text data to numerical format using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)  # Convert text to numerical features

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_vectorized, y)

# Convert the sparse matrix back to a DataFrame
balanced_df_vectorized = pd.DataFrame(X_balanced.toarray(), columns=vectorizer.get_feature_names_out())
balanced_df_vectorized["Label"] = y_balanced  # Add labels column

# Save the balanced dataset after vectorization
balanced_df_vectorized.to_csv("balanced_instagram_data_vectorized.csv", index=False)
print("\n✅ Balanced dataset (vectorized) saved as 'balanced_instagram_data_vectorized.csv'")





