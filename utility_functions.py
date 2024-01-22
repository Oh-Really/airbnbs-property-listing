from sklearn import model_selection
from pathlib import Path
import os
import joblib
import json

def get_training_validation_data(features, labels):
    '''Returns the train, validation and test data sets'''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=20)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
    X_test, y_test, test_size=0.5, random_state=20)
    return (X_train, X_validation, X_test, y_train, y_validation, y_test)


def save_model(folder: str, sub_folder: str, model_class, hyperparams_grid, tune_hyperparam_func):
    '''Saves the best model from the function passed in as tune_hyperparam_func argument to desired 
    folder, along with json of best performing hyperparms and corresponding performance metrics for that model'''

    model, hyperparams, metrics = tune_hyperparam_func(model_class, hyperparams_grid)
    
    output_folder = Path(os.getcwd(), f'models/{folder}', sub_folder)

    hyperparams = json.dumps(hyperparams)
    file_path = os.path.join(output_folder, "hyperparameters.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as json_file:
        json_file.write(hyperparams)
    
    
    metrics = json.dumps(metrics)
    file_path = os.path.join(output_folder, "metrics.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as json_file:
        json_file.write(metrics)

    file_path = os.path.join(output_folder, "model.joblib")
    joblib.dump(model, file_path)
