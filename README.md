# NLP Healthcare Analysis

This project performs Natural Language Processing (NLP) analysis on a healthcare dataset from Kaggle. The goal is to build a sophisticated model that can accurately predict test results based on a patient's medical condition and prescribed medication.

## Features

*   **Advanced Text Preprocessing:** Utilizes the `nltk` library for tokenization, stop-word removal, and stemming to create meaningful features from the text data.
*   **Powerful XGBoost Model:** Employs the `XGBoost` classifier, a high-performance gradient boosting algorithm, for accurate predictions.
*   **Hyperparameter Tuning:** Uses `GridSearchCV` to find the optimal hyperparameters for the XGBoost model, maximizing its performance.
*   **Modular and Organized Code:** The project is structured with separate modules for data preprocessing, model training, and prediction.
*   **Interactive Frontend:** A React-based web interface for easy interaction with the prediction model.

## Dataset

The dataset used in this project is the [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) from Kaggle. It contains information about patient demographics, medical conditions, treatments, and outcomes.

**Please Note:** You will need to download the dataset from the Kaggle link above and place it in a `data` directory within this project.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/nlp-healthcare-analysis.git
    cd nlp-healthcare-analysis
    ```

2.  **Create a virtual environment (for backend):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the data:**
    Download the dataset from [Kaggle](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) and place the `healthcare_dataset.csv` file in a `data` directory.

5.  **Install frontend dependencies:**
    ```bash
    cd frontend
    npm install
    cd ..
    ```

## Usage

### Training the Model (Backend)

To train the model, run the following command from the project root:

```bash
python src/main.py
```

This will perform the following steps:

1.  Load and preprocess the data.
2.  Perform hyperparameter tuning using `GridSearchCV` to find the best model.
3.  Train the final model on the entire training set.
4.  Save the trained model and the label encoder to the `models` directory.

### Running the Backend API

To start the FastAPI backend server, run the following command from the project root:

```bash
uvicorn api:app --reload
```

This will typically start the API on `http://127.0.0.1:8000`.

### Running the Frontend

To start the React frontend development server, open a **new terminal** and run the following commands:

```bash
cd frontend
npm start
```

This will typically open the frontend in your browser at `http://localhost:3000`.

### Making Predictions (via Frontend)

Once both the backend API and the frontend are running, you can interact with the model through the web interface. Fill in the patient information in the form and click "Get Prediction" to see the predicted test result.

## Project Structure

```
.
├── data
│   └── healthcare_dataset.csv
├── models
│   ├── nlp_model.pkl
│   └── label_encoder.pkl
├── src
│   ├── __init__.py
│   ├── main.py
│   └── preprocess.py
├── frontend
│   ├── public
│   ├── src
│   │   ├── App.css
│   │   ├── App.js
│   │   ├── index.css
│   │   └── index.js
│   ├── package.json
│   └── ... (other React files)
├── .gitignore
├── README.md
└── requirements.txt
```
