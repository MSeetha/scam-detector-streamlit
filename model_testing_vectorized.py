import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
X_test, y_test = joblib.load("test_data.pkl")

# Load trained model
model = joblib.load("scam_detection_model_vectorized.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Save predictions
test_results = pd.DataFrame({"Actual_Label": y_test, "Predicted_Label": y_pred})
test_results.to_csv("test_predictions_vectorized.csv", index=False)

print("✅ Model testing completed. Predictions saved to 'test_predictions_vectorized.csv'")
