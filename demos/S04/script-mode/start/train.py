# Imports
import argparse, os
import boto3
import json
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

if __name__ == '__main__':
    
    # Retrieve the hyperparameters and arguments passed to the script
    parser = argparse.ArgumentParser()
    
    # Hyperparamaters
    parser.add_argument('--estimators', type=int, default=15)
    
    # Directories to write model and output artifacts to after training
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()
    estimators     = args.estimators
    model_dir  = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir   = args.train
    
    
    # Read in the training data
    df = pd.read_csv(training_dir + '/train.csv', sep=',')
    
    # Data preprocessing to pull out target column ('quality') and split into training and testing sets
    X = df.drop('quality', axis = 1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train) 
    X_test = sc.transform(X_test)
    
    
     # Train the model using Random Forest Regression
    regressor = RandomForestRegressor(n_estimators=estimators)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    # Save the model
    joblib.dump(regressor, os.path.join(args.sm_model_dir, "model.joblib"))
    
    
# INFERENCE
# SageMaker uses four functions to load the model and use it for inference: model_fn, input_fn, output_fn, and predict_fn
    
"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        request_body = json.loads(request_body)
        inpVar = request_body['Input']
        return inpVar
    else:
        raise ValueError("This model only supports application/json input")

"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
"""

def output_fn(prediction, content_type):
    res = int(prediction[0])
    respJSON = {'Output': res}
    return respJSON