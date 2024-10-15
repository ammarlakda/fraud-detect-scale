import sagemaker
import boto3
from sagemaker import Session
from sagemaker.sklearn import SKLearnModel

# Manually specify your SageMaker role (replace with the actual ARN of your role)
role = 'arn:aws:iam::381492199311:role/service-role/AmazonSageMaker-ExecutionRole-20241014T231402'

# Specify the S3 paths for the model and inference script
model_s3_path = 's3://sagemaker-studio-381492199311-kxyl9vd5y3k/models/model.tar.gz'

# Initialize a SageMaker session (ensure the region is supported)
sagemaker_session = Session(boto_session=boto3.Session(region_name="us-east-1"))

# Create and deploy the model
model = SKLearnModel(
    model_data=model_s3_path,
    role=role,
    entry_point='notebooks/inference.py',
    framework_version='0.23-1',  # Specify the Scikit-learn version
    sagemaker_session=sagemaker_session
)

# Deploy the model to a SageMaker endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'  # Adjust instance type based on your requirements
)

# Test the deployed model
input_data = {"features": [1.2, 0.4, -1.0, 3.5, 2.1, -0.7,1.2, 0.4, -1.0, 3.5, 2.1, -0.7,1.2, 0.4, -1.0, 3.5, 2.1, -0.7,1.2, 0.4, -1.0, 3.5, 2.1, -0.7,1.2, 0.4, -1.0, 3.5, 2.1, -0.7]}  # Example input
result = predictor.predict(input_data)
print(result)
