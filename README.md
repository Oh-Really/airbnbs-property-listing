### Modelling Airbnb'a Property dataset

In this project, we aim to use a dataset of Airbnb properties along with accompanying images to build, train, tune and evaluate models on that dataset.

The project is split into a few parts, namely;
- **Data cleaning and handling**. Here we evaluate the input airbnb data and clean it by removing certain rows with null values, cleaning the descriptions column into a readable string, as well as selecting only numerical parts of the whole dataframe.
- **Creating a Baseline model**. During this step, we create a Linear Regression model and evaluate its performance on our validation set using metrics such as RMSE and $R^2$. I the classification case, this is done with a Logistic Regression model evaluating accuracy, recall, and the F1 score instead.
- **Evaluating the models**. Here we evaluate the models performance against the criterion of our choosing.
- **Optimising the model**. Finally, we tune the model by cycling through different hyperparameter combinations using GridSearchCV, and also differnt models as well (Decision Tree's, Random Forests etc.) Then we can assess which model gives us the best performance overall.

## Data Preparation
The first step is to read in the data from the listings.csv file, and clean any necessary parts of the original data.

### Data Cleaning ###
Here, we perform data cleaning on our source data. Firstly, we load the csv file as a DataFrame to then explore the dataset for any missing values, rows with null values, or data that looks incorrect/out of place. In our case, rows were removed when given columns were missing values. Other columns that had missing values we set defaults in their place, and the description column had it's string representations of the description of a property cleaned up.

## Regression models
The first set of models that we used in this project were regression models, which are used to predict a continuous value, such as the Price_Night column of our dataset.

For such models, we typically want to minimise, or optimise, a particular loss funcion/parameter. One way of doing this is via gradient descent. A particular variant of this is Stochastic Gradient Descent.

### Stochastic Gradient Descent
