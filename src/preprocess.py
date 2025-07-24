import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

# Download NLTK data (if not already downloaded)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Cleans and preprocesses a single text string."""
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove non-letters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lower case
    text = text.lower()
    # Tokenize
    words = text.split()
    # Remove stopwords
    words = [w for w in words if not w in stopwords.words('english')]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    # Join the words back into one string
    return " ".join(words)

def preprocess_data(filepath):
    """Loads and preprocesses the healthcare data."""
    df = pd.read_csv(filepath)
    # For this example, let's assume we're predicting 'Test Results'
    # from 'Medical Condition' and 'Medication'.
    # We'll combine these text fields into a single feature.
    df['text_features'] = df['Medical Condition'] + ' ' + df['Medication']
    df['text_features'] = df['text_features'].apply(preprocess_text)
    return df
