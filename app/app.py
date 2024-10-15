from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('models/fraud_detection_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert data into the correct format for the model
    features = np.array([data['features']])
    # Make a prediction
    prediction = model.predict(features)
    # Return the prediction as a response
    return jsonify({'fraud_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)