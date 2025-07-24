import React, { useState } from 'react';
import './App.css';

function App() {
    const [age, setAge] = useState('');
    const [gender, setGender] = useState('Male');
    const [bloodType, setBloodType] = useState('O+');
    const [medicalCondition, setMedicalCondition] = useState('');
    const [medication, setMedication] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);

    const getPrediction = async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    age: parseInt(age),
                    gender,
                    blood_type: bloodType,
                    medical_condition: medicalCondition,
                    medication,
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to get prediction');
            }

            const data = await response.json();
            setPrediction(data.predicted_test_result);
            setError(null);
        } catch (err) {
            setError(err.message);
            setPrediction(null);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Healthcare Prediction</h1>
                <div className="form-container">
                    <input
                        type="number"
                        placeholder="Age"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                    />
                    <select value={gender} onChange={(e) => setGender(e.target.value)}>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                    <select value={bloodType} onChange={(e) => setBloodType(e.target.value)}>
                        <option value="A+">A+</option>
                        <option value="A-">A-</option>
                        <option value="B+">B+</option>
                        <option value="B-">B-</option>
                        <option value="AB+">AB+</option>
                        <option value="AB-">AB-</option>
                        <option value="O+">O+</option>
                        <option value="O-">O-</option>
                    </select>
                    <textarea
                        placeholder="Medical Condition"
                        value={medicalCondition}
                        onChange={(e) => setMedicalCondition(e.target.value)}
                    />
                    <textarea
                        placeholder="Medication"
                        value={medication}
                        onChange={(e) => setMedication(e.target.value)}
                    />
                    <button onClick={getPrediction}>Get Prediction</button>
                </div>
                {prediction && (
                    <div className="prediction-result">
                        <h2>Predicted Test Result:</h2>
                        <p>{prediction}</p>
                    </div>
                )}
                {error && (
                    <div className="error-message">
                        <h2>Error:</h2>
                        <p>{error}</p>
                    </div>
                )}
            </header>
        </div>
    );
}

export default App;