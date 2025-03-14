import pandas as pd
import joblib

# Load test dataset
df_test = pd.read_csv("X_test.csv")

# Load TF-IDF vectorizer and transform test data
vectorizer = joblib.load("tfidf_vectorizer.pkl")
X_test_vectorized = vectorizer.transform(df_test["Captions"])  # Change column name

# Load trained model
model = joblib.load("scam_detection_model.pkl")

# Make predictions
predictions = model.predict(X_test_vectorized)

# Save predictions
df_test["Predicted_Label"] = predictions
df_test.to_csv("test_predictions.csv", index=False)

print("✅ Model testing completed. Predictions saved to test_predictions.csv")

