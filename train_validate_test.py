import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load the balanced dataset
file_path = "balanced_instagram_data_vectorized.csv"
df = pd.read_csv(file_path)

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Split dataset (80% train, 10% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True)
}

# Train and validate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    
    # Evaluate validation performance
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, pos_label="Scam")
    recall = recall_score(y_val, y_val_pred, pos_label="Scam")
    f1 = f1_score(y_val, y_val_pred, pos_label="Scam")
    
    print(f"✅ {model_name} Validation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-score: {f1:.4f}\n")
    
    # Save the trained model
    model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"✅ {model_name} model saved as {model_filename}\n")

# Test best model (example: Random Forest)
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

# Evaluate test performance
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, pos_label="Scam")
recall = recall_score(y_test, y_test_pred, pos_label="Scam")
f1 = f1_score(y_test, y_test_pred, pos_label="Scam")

print(f"✅ Test Results for Best Model (Random Forest):")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1-score: {f1:.4f}\n")
