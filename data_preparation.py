import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Download necessary resources for text processing
nltk.download('punkt')
nltk.download('stopwords')

# Load cleaned data
df = pd.read_csv("cleaned_instagram_data.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Convert text to lowercase
df['Captions'] = df['Captions'].str.lower()

# Remove special characters, numbers, and punctuation
df['Captions'] = df['Captions'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

# Tokenize text
df['Captions'] = df['Captions'].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['Captions'] = df['Captions'].apply(lambda x: [word for word in x if word not in stop_words])

print("Data Cleaning Completed!")
print(df.head())

# Save the processed dataset
df.to_csv("prepared_instagram_data.csv", index=False)
print("Prepared data saved as 'prepared_instagram_data.csv'")
