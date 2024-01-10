# %%
from get_data import get_training_validation_data, save_model
from tabular_data import load_data, load_airbnb
from sklearn import model_selection
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

grid = {
    'multi_class' : ['ovr', 'multinomial'],
    'penalty': ['l2', 'none'],
    'max_iter' : [100, 1000, 1000]
 }

#print(tune_classification_model_hyperparameters(LogisticRegression, grid))
save_model('classification', 'linear_regression', LogisticRegression, grid, tune_classification_model_hyperparameters)



# %%
