1. This is a K-nearest neighbor classifier trained on face image dataset by fetching them from AWS S3 bucket using pre-built functions from global_functions which utilizes boto3.
2. The trained model is pushed back to S3 bucket using pre-built functions from global_functions which utilizes boto3.
3. The model is tested by fetching the model from S3 and passing a PIL image object for class prediction, the predictions are recorded in a dataframe which is also pushed to the S3 bucket for the model's performance analysis and hyper-parameter tuning
