import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("prepared_instagram_data.csv")  # Change to your actual dataset file

# Separate features and labels
X = df["Captions"]  # Change to your actual text column name
y = df["Label"]

# Convert text data to numerical using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust features as needed
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Save vectorizer for later use
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Apply SMOTE (only on training data)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Save trained model
joblib.dump(model, "scam_detection_model.pkl")

print("✅ Model training completed and saved as scam_detection_model.pkl")





