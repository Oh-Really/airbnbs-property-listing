# %%
from utility_functions import get_training_validation_data, save_model
from tabular_data import load_data, load_airbnb
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import numpy as np
import joblib
import os
import json


airbnb_df = load_data('clean_tabular_data.csv')
airbnb_df.drop(columns=airbnb_df.columns[0], axis=1, inplace=True)
features, labels = load_airbnb(airbnb_df, "Price_Night")

X_train, X_validation, X_test, y_train, y_validation, y_test = get_training_validation_data(features, labels)


# %%
# Tuning hyperparameters is the main reason we need the validation set. Each permutation of 
# hyperparams should be tested on the same validation set

def grid_search(hyperparameters: dict):
    '''Returns Generator object that gives all permutations of input hyperparamaters  '''
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))


def custom_tune_regression_model_hyperparameters(model_class, hyperparams_grid: dict, *sets):
    ''' Custom grid search function'''
    best_hyperparams, best_loss = None, np.inf

    for hyperparams in grid_search(hyperparams_grid):
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)

        y_validation_pred = model.predict(X_validation)
        validation_RMSE = mean_squared_error(y_validation, y_validation_pred, squared=False)

        if validation_RMSE < best_loss:
            best_loss = validation_RMSE
            best_hyperparams = hyperparams

    performance_metric = {'validation_RMSE': best_loss}
    print(f"Best loss: {best_loss}")
    print(f"Best hyperparameters: {best_hyperparams}")

    return model_class, best_hyperparams, performance_metric

# grid = {
#     "loss": ['squared_error', 'huber'],
#     "max_iter": [1000, 10000, 100000],
#     "learning_rate": ['constant', 'optimal', 'invscaling']
# }
#custom_tune_regression_model_hyperparameters(SGDRegressor, grid, X_validation, X_test, X_train, y_validation, y_test, y_train)
# %%
def tune_regression_model_hyperparameters(model_class, hyperparams_grid):
    ''' 
    Function to tune hyperparameters using GridSearch

    Parameters
    ---------
    model_class: Model to be trained i.e. SGDRegressor
    hyperparams_grid: Grid of hyperparameters which GridSearch will create permutations out of

    Returns
    -------
    best_model: The best performing model
    best_hyperparams: Hyperparamater set for the best performing model
    results_dict: RMSE and R^2 score of the best model on the validation set   
    '''
    model = model_class(random_state=0)
    search = model_selection.GridSearchCV(estimator=model, param_grid=hyperparams_grid, scoring='neg_root_mean_squared_error')
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    train_RMSE = np.abs(search.best_score_) # Train RMSE - don't necessarily need this

    y_val_predict = best_model.predict(X_validation) #Validation predection
    validation_RMSE = mean_squared_error(y_validation, y_val_predict, squared=False)
    validation_r2 = r2_score(y_validation, y_val_predict)    


    results_dict = {'validation_RMSE': validation_RMSE, 'validation_r2': validation_r2}
    best_hyperparams = search.best_params_
    return best_model, best_hyperparams, results_dict


#clf = tune_regression_model_hyperparameters(SGDRegressor, grid)
# %%
def evaluate_different_models():
    '''
    Function to iterate over different model classes and hyperparameter grids to evaluate best performing model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
 
    
    SGDRegressor_hyperparameters = {"loss": ['squared_error', 'huber'],
    "max_iter": [1000, 10000, 100000],
    "learning_rate": ['constant', 'optimal', 'invscaling'],
    "penalty": ['l2', 'l1', 'elasticnet'],
    "alpha": [1e-3,1e-4,1e-5]
    }
    decisiontree_hyperparameters = {"min_samples_leaf": [2, 4, 6],
        "min_samples_split" : [1, 2, 3, 4],
        "splitter": ['best','random'],
        "criterion": ['squared_error','friedman_mse','absolute_error','poisson'],
        "max_features": ['auto','sqrt','log2']
    }
    randomforest_hyperparameters = {
        'criterion': ['squared_error','absolute_error','poisson'],
        'max_depth' : [10,100,1000],
        'min_samples_leaf' : [1, 10, 50]
    }
    gradientboost_hyperparameters = {
        'loss': ['squared_error','absolute_error','huber','quantile'],
        'min_weight_fraction_leaf': [0.0,1e-5,1e-3],
        'max_features': ['auto','sqrt','log2']
    }

    model_list = [SGDRegressor,DecisionTreeRegressor,RandomForestRegressor,GradientBoostingRegressor]
    hyperparam_list = [SGDRegressor_hyperparameters, decisiontree_hyperparameters, randomforest_hyperparameters, gradientboost_hyperparameters]    
    folder_list = ['linear_regression', 'decision_tree', 'random_forest', 'gradient_boost']

    for (sub_folder, model, param_grid) in zip(folder_list, model_list, hyperparam_list):
        try:
            save_model('regression', sub_folder, model, param_grid, tune_regression_model_hyperparameters)
        except:
            pass

    return

def find_best_model():
    '''
    First, obtain a list of the absolute paths for all metrics.json files.
    Then, open these up in a loop, compare the Validation_RMSE score to previous. If better than previous score, save model in best_model var
    to be returned, along with that models hyperparameters.
    '''
    # Get a list of absolute paths
    folder_list = os.listdir(Path(os.getcwd(), 'airbnb/models/regression'))
    i = 0
    for folder in folder_list:
        folder_list[i] = os.path.join(Path(os.getcwd(), 'airbnb/models/regression'), folder, 'metrics.json')
        i += 1
    
    #for every metrics.json file in the models/regression folder:
    #open the metrics.json file, check the validation_RMSE score and save that and model if better than previous
    best_score = np.inf
    for files in folder_list:
        with open(files, 'r') as data_file:
            data = json.load(data_file)
            score = data['validation_RMSE']
            if score < best_score:
                best_score = score
    
    # TODO return model, hyperparams_grid, and metrics for best model

    


if __name__ == "__main__":
    # grid = {
    # "loss": ['squared_error', 'huber'],
    # "max_iter": [1000, 10000, 100000],
    # "learning_rate": ['constant', 'optimal', 'invscaling'],
    # "penalty": ['l2', 'l1', 'elasticnet'],
    # "alpha": [1e-3,1e-4,1e-5]
    # }
    #save_model('regression\linear_regression', SGDRegressor, grid. tune_regression_model_hyperparameters)
    evaluate_different_models()