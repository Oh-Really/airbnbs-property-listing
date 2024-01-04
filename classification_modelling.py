# %%
from tabular_data import load_data, load_airbnb
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


airbnb_df = load_data('clean_tabular_data.csv')
airbnb_df.drop(columns=airbnb_df.columns[0], axis=1, inplace=True)
features, labels = load_airbnb(airbnb_df, "Category")

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=20)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
X_test, y_test, test_size=0.5, random_state=20)


# %%
def my_Log_Regression():
    clf_model = LogisticRegression(max_iter = 100000)
    clf_model.fit(X_train, y_train)
    prediction_value = clf_model.predict(X_test)

    cf = confusion_matrix(y_test, prediction_value)
    print(f'Confusion matrix, \n \n {cf} \n')

    precision_value = precision_score(prediction,y_test,average='weighted',zero_division=0)
    print(f'Precision Score: {precision_value} \n')

    recall_value = recall_score(prediction,y_test,average='weighted',zero_division=0)
    print(f'Recall Score: {recall_value} \n')

    f1_value = f1_score(prediction,y_test,average='weighted')
    print(f'F1 Score: {f1_value} \n')

    model_score = model.score(X_test,y_test)
    print(f'Mean Accuracy score: {model_score}') #Mean Accuracy score, in interval [0,1] (In this case, roughly 42% correctly predicted. Not too bad considering we have 5 different types of property to predict)
    return
# %%
