# %%
#import sys
#print(sys.path)
from tabular_data import load_data, load_airbnb
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd


# Load the data into a dataframe and Series
airbnb_df = load_data('clean_tabular_data.csv')
airbnb_df.drop(columns=airbnb_df.columns[0], axis=1, inplace=True)
features, labels = load_airbnb(airbnb_df, "Price_Night")

# Split the data into a training, validation, and test sets for both X and y
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
X_test, y_test, test_size=0.5)

model = SGDRegressor()
model.fit(X_train, y_train)

# Make predictions of my target based off features
y_train_pred = model.predict(X_train)
y_validation_pred = model.predict(X_validation)
y_test_pred = model.predict(X_test)

# Print out evalution metrics
training_loss_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_loss_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
training_loss_r2 = r2_score(y_train, y_train_pred)
test_loss_r2 = r2_score(y_test, y_test_pred)
validation_loss_rmse = mean_squared_error(y_validation, y_validation_pred, squared=False)

print(f"Training Loss for RMSE:  {training_loss_rmse}\n"
      f"Test Loss for RMSE:  {training_loss_rmse}\n"
      f"Training Loss for R2:  {training_loss_r2}\n"
      f"Test Loss for R2  {training_loss_r2}\n"
      f"Validation Loss for RMSE:  {validation_loss_rmse}")

#plt.plot(y_train_pred, y_train)
#plt.show()



# %%
# Tuning hyperparameters is the main reason we need the validation set. Each permutation of 
# hyperparams should be tested on the same validation set

# Returns Generator object that gives all permutations of input hyperparamaters
def grid_search(hyperparameters: dict):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def custom_tune_regression_model_hyperparameters(model_class, hyperparams_grid: dict, *sets):
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

grid = {
    "loss": ['squared_error', 'huber'],
    "max_iter": [1000, 10000, 100000],
    "learning_rate": ['constant', 'optimal', 'invscaling']
}

custom_tune_regression_model_hyperparameters(SGDRegressor, grid, X_validation, X_test, X_train, y_validation, y_test, y_train)
# %%
grid = {
    "loss": ['squared_error', 'huber'],
    "max_iter": [1000, 10000, 100000],
    "learning_rate": ['constant', 'optimal', 'invscaling'],
    "penalty": ['l2', 'l1', 'elasticnet']
}
def tune_regression_model_hyperparameters(model_class, hyperparams_grid):
    model = model_class(random_state=0)
    search = model_selection.GridSearchCV(estimator=model, param_grid=hyperparams_grid, scoring='neg_root_mean_squared_error')
    search.fit(X_train, y_train)
    y_validation_pred = search.predict(X_validation)
    validation_RMSE = mean_squared_error(y_validation, y_validation_pred, squared=False)
    print(validation_RMSE)

    a = search.best_estimator_
    results_df = pd.DataFrame(search.cv_results_) #Puts results into df
    results_df = results_df.sort_values(by=["rank_test_score"])
    params = search.best_params_
    index = search.best_index_
    best_row = results_df.iloc[index]


clf = tune_regression_model_hyperparameters(SGDRegressor, grid)
# %%
