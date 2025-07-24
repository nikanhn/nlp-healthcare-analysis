import pandas as pd
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pickle

def main():
    """Main function to run the NLP analysis."""
    # Load and preprocess data
    df = preprocess_data('data/healthcare_dataset.csv')

    # Encode the target variable
    le = LabelEncoder()
    df['Test Results'] = le.fit_transform(df['Test Results'])

    # Split data into training and testing sets
    X = df[['Age', 'Gender', 'Blood Type', 'text_features']]
    y = df['Test Results']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'text_features'),
            ('categorical', 'passthrough', ['Age', 'Gender', 'Blood Type'])
        ])

    # Create the model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ])

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1, 0.01]
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Save the model and label encoder
    with open('models/nlp_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("Model trained and saved to models/nlp_model.pkl")
    print(f"Best parameters found: {grid_search.best_params_}")

    # Evaluate the model
    from sklearn.metrics import classification_report
    y_pred = best_model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == '__main__':
    main()
