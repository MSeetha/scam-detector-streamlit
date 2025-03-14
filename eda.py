import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Dataset
file_path = os.path.join(os.getcwd(), 'instagram_dataset', 'prepared_instagram_data.csv')


try:
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found!")

    print("\n📂 File found! Loading dataset...")

    # Load Dataset
    df = pd.read_csv(file_path)

    # Display first few rows
    print("\n📊 Dataset Sample:")
    print(df.head())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\n🔍 Missing Values:")
    print(missing_values)

    # Label Distribution
    print("\n📊 Visualizing Label Distribution...")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Label', data=df, palette='coolwarm', hue='Label', legend=False)
    plt.title("Label Distribution")
    plt.show()

    # ✅ Convert Captions from stringified lists to normal strings
    print("\n🔄 Converting tokenized captions into strings...")
    df['Processed_Captions'] = df['Captions'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else "")

    print("\n🛠 Applying TF-IDF Vectorization...")
    
    # ✅ Apply TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['Processed_Captions'])

    print("✅ TF-IDF Transformation Complete!")

    # ✅ Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

    print("\n📊 TF-IDF Feature Matrix Shape:", tfidf_df.shape)

    # ✅ Save Extracted Features
    tfidf_output_path = os.path.join(os.getcwd(), "tfidf_features.csv")
    tfidf_df.to_csv(tfidf_output_path, index=False)
    print(f"✅ TF-IDF features saved successfully as: {tfidf_output_path}")

except Exception as e:
    print("\n❌ An error occurred:", str(e))

