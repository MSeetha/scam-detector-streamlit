import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load vectorized balanced dataset
df = pd.read_csv("balanced_instagram_data_vectorized.csv")
X = df.drop(columns=["Label"])
y = df["Label"]

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define hyperparameter grids
param_grid_lr = {"C": [0.01, 0.1, 1, 10, 100]}
param_grid_rf = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
param_grid_svm = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}

# Fine-tune Logistic Regression
lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5)
lr.fit(X_train, y_train)
print("✅ Best Logistic Regression Params:", lr.best_params_)

# Fine-tune Random Forest
rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
rf.fit(X_train, y_train)
print("✅ Best Random Forest Params:", rf.best_params_)

# Fine-tune SVM
svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
svm.fit(X_train, y_train)
print("✅ Best SVM Params:", svm.best_params_)

# Evaluate best models
best_lr = lr.best_estimator_
best_rf = rf.best_estimator_
best_svm = svm.best_estimator_

models = {"Logistic Regression": best_lr, "Random Forest": best_rf, "SVM": best_svm}

for name, model in models.items():
    y_pred = model.predict(X_val)
    print(f"\n✅ {name} Validation Results:")
    print(f"   Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_val, y_pred, pos_label='Scam'):.4f}")
    print(f"   Recall: {recall_score(y_val, y_pred, pos_label='Scam'):.4f}")
    print(f"   F1-score: {f1_score(y_val, y_pred, pos_label='Scam'):.4f}")
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_best.pkl")  # Save best models
