import pandas as pd
import joblib
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load the vectorized dataset
df = pd.read_csv("balanced_instagram_data_vectorized.csv")

# Separate features and labels
X = csr_matrix(df.drop(columns=["Label"]).values)  # Convert to sparse matrix
y = df["Label"]

# Split data: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and datasets
joblib.dump(model, "scam_detection_model_vectorized.pkl")
joblib.dump((X_train, y_train), "train_data.pkl")
joblib.dump((X_val, y_val), "val_data.pkl")
joblib.dump((X_test, y_test), "test_data.pkl")

print("✅ Model trained and saved as 'scam_detection_model_vectorized.pkl'")
print("✅ Training, validation, and test data saved.")
