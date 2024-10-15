import io
import os
import joblib
import json
import numpy as np

def model_fn(model_dir):
    """Load the model for inference"""
    model = joblib.load(os.path.join(model_dir, "fraud_detection_model.pkl"))  # Adjust the file name based on your tarball
    return model

def input_fn(request_body, content_type='application/json'):
    """Deserialize the input data"""
    
    if content_type == 'application/json':
        # Assuming the input is JSON and contains a list of features
        data = json.loads(request_body)
        # Convert to a NumPy array
        return np.array(data['features']).reshape(1, -1)
    
    elif content_type == 'application/x-npy':
        # Handle NumPy array format
        stream = io.BytesIO(request_body)
        return np.load(stream, allow_pickle=True)
    
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make predictions using the model"""
    return model.predict(input_data)

def output_fn(prediction, content_type='application/json'):
    """Serialize the prediction output"""
    return {'prediction': prediction.tolist()}
