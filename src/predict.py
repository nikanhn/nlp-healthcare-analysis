import pickle

def predict(medical_condition, medication):
    """Loads the model and makes a prediction."""
    # Load the trained model
    with open('models/nlp_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Combine the input features
    text_features = medical_condition + ' ' + medication

    # Make a prediction
    prediction = model.predict([text_features])

    return prediction[0]

if __name__ == '__main__':
    # Get user input
    medical_condition = input("Enter the medical condition: ")
    medication = input("Enter the medication: ")

    # Get the prediction
    result = predict(medical_condition, medication)

    print(f"\nPredicted Test Result: {result}")

