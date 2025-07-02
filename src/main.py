from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

def main():
    """Main function to run the NLP analysis."""
    # Load and preprocess data
    df = preprocess_data('data/healthcare_dataset.csv')

    # For this example, let's assume we're predicting 'Test Results' 
    # from 'Symptoms' and 'Medical Condition'. 
    # We'll combine these text fields into a single feature.
    df['text_features'] = df['Symptoms'] + ' ' + df['Medical Condition']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_features'], df['Test Results'], test_size=0.2, random_state=42
    )

    # Create and train the NLP model
    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    model.fit(X_train, y_train)

    # Save the model
    with open('models/nlp_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved to models/nlp_model.pkl")

if __name__ == '__main__':
    main()
