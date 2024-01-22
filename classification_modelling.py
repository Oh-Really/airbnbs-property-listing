# %%
from utility_functions import get_training_validation_data, save_model
from tabular_data import load_data, load_airbnb
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


airbnb_df = load_data('clean_tabular_data.csv')
airbnb_df.drop(columns=airbnb_df.columns[0], axis=1, inplace=True)
features, labels = load_airbnb(airbnb_df, "Category")

# We see that the labels now have 5 unique, nominal categories. As such, for any Logistic Regression model we intend to implement,
# we need to encode these labels and ensure the Logistic Regression model is a multinomial model.

# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)

X_train, X_validation, X_test, y_train, y_validation, y_test = get_training_validation_data(features, labels)
# %%
def my_Log_Regression():
    clf_model = LogisticRegression(max_iter = 100000)
    clf_model.fit(X_train, y_train)
    prediction_value = clf_model.predict(X_test)

    cf = confusion_matrix(y_test, prediction_value)
    print(f'Confusion matrix, \n \n {cf} \n')

    precision_value = precision_score(prediction_value,y_test,average='weighted',zero_division=0)
    print(f'Precision Score: {precision_value} \n')

    recall_value = recall_score(prediction_value,y_test,average='weighted',zero_division=0)
    print(f'Recall Score: {recall_value} \n')

    f1_value = f1_score(prediction_value,y_test,average='weighted')
    print(f'F1 Score: {f1_value} \n')

    accuracy = clf_model.score(X_test,y_test)
    print(f'Accuracy: {accuracy} \n')
    return

def tune_classification_model_hyperparameters(model_class, hyperparams_grid):
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_training_validation_data(features, labels)
    gridsearch = model_selection.GridSearchCV(estimator=model_class(random_state = 0), param_grid=hyperparams_grid)

    gridsearch.fit(X_train, y_train)
    best_model = gridsearch.best_estimator_
    y_val_prediction = best_model.predict(X_validation)

    precision_value = precision_score(y_val_prediction, y_validation,average='weighted', zero_division=0)
    recall_value = recall_score(y_val_prediction, y_validation, average='weighted', zero_division=0)
    f1_value = f1_score(y_val_prediction, y_validation, average='weighted')
    accuracy_score = best_model.score(X_validation, y_validation)
    print(accuracy_score)

    best_hyperparams = gridsearch.best_params_
    results_dict = {
        "validation_accuracy" : accuracy_score,
        "precision_score" : precision_value,
        "recall_score" : recall_value,
        "F1_score" : f1_value
    }

    return best_model, best_hyperparams, results_dict

def evaluate_different_models():
    '''
    Setup a series of classification models and their hyperparamter grids. These models will then be passed to 
    tune_classification_model_hyperparameters method to assess and record their performances.
    '''
    logistic_regression_hyperparameters = {
    'multi_class' : ['ovr', 'multinomial'],
    'penalty': ['l2', 'none'],
    'max_iter' : [100, 1000, 1000]
    }
    decisiontree_hyperparameters = {
        "criterion": ['gini', 'entropy', 'log_loss'],
        "max_features" : ['auto', 'sqrt', 'log2'],
        "splitter": ['best','random']
    }
    random_forest_hyperparameters = {
        "criterion": ['gini', 'entropy', 'log_loss'],
        "max_features" : ['auto', 'sqrt', None],
        "min_samples_leaf": [1,2,3]
    }
    gradientboosting_hyperparameters = {
        'loss': ['log_loss'], 
        'criterion': ['friedman_mse','squared_error'],
        'max_features': [1,'sqrt','log2']
    }

    model_list = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier]
    hyperparam_list = [logistic_regression_hyperparameters, decisiontree_hyperparameters, random_forest_hyperparameters, gradientboosting_hyperparameters]    
    folder_list = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boost']

    for (sub_folder, model, param_grid) in zip(folder_list, model_list, hyperparam_list):
        try:
            save_model('classification', sub_folder, model, param_grid, tune_classification_model_hyperparameters)
        except:
            pass

    return


if __name__ == "__main__":
    evaluate_different_models()

