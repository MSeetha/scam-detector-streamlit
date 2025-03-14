import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load validation & test data
X_val, y_val = joblib.load("val_data.pkl")
X_test, y_test = joblib.load("test_data.pkl")

# Load trained model
model = joblib.load("scam_detection_model_vectorized.pkl")

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, pos_label="Scam")
val_recall = recall_score(y_val, y_val_pred, pos_label="Scam")
val_f1 = f1_score(y_val, y_val_pred, pos_label="Scam")

print("\n📊 Validation Set Performance:")
print(f"✅ Accuracy: {val_acc:.4f}")
print(f"✅ Precision: {val_precision:.4f}")
print(f"✅ Recall: {val_recall:.4f}")
print(f"✅ F1-score: {val_f1:.4f}")

# Evaluate on test set
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, pos_label="Scam")
test_recall = recall_score(y_test, y_test_pred, pos_label="Scam")
test_f1 = f1_score(y_test, y_test_pred, pos_label="Scam")

print("\n📊 Test Set Performance:")
print(f"✅ Accuracy: {test_acc:.4f}")
print(f"✅ Precision: {test_precision:.4f}")
print(f"✅ Recall: {test_recall:.4f}")
print(f"✅ F1-score: {test_f1:.4f}")
