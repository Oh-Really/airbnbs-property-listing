# Modelling Airbnb's Property dataset

In this project, we aim to use a dataset of Airbnb properties along with accompanying images to build, train, tune, and evaluate models on that dataset.

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
For a linear model, we make a prediction, denoted by $\widetilde{y}$, of our feature. The equation for this is:

$$
\widetilde{y} = h(x) = x \cdot w_0+b
$$

In this equation $w$ represents the weight vector, which is a vector containing the weights for every input in the parameter vector $x$, and $b$ is a bias value. In linear models, this can be linked to the intercept and slope. The loss function can be written as:

$$
L = \frac{1}{N}\sum_{i=0}^{N-1}(y_i-\widetilde{y}_i)^2
$$

This calculates the sum of the residuals for each observation and our estimated value. We want to take the derivative of this function with respect to the parameters, in our case our weights. We can do so via:

$$
w_{i+1} = w_i - \eta \nabla L.
$$

$\eta$ here denotes the learning rate which dictates how much we change our weights by. We are updating the parameters, $w$, in the opposite direction of the gradient of the loss function w.r.t. the parameters, hence the minus sign. The new weight vector is substituted back into our prediction $\widetilde{y}$ giving us a new model, which we then use when taking the derivative of the loss function with the updated parameters. We repeat this process until the weight vector converges to an optimal vector

$$
w_i \longrightarrow{} w^*.
$$

### RMSE and $R^2$
RMSE is a certain type of loss function commonly used in regression problems and measures the distance between a predicted value and the actual value. For a given prediction, $\widetilde{y}_i$, and a real value given by ${y}_i$, the RMSE is measured by

$$
RMSE = \sqrt{\sum_{i=1}^{N}\frac{(\widetilde{y}_i-y_i)^2}{N}},
$$

where $N$ is the total number of data points.

$R^2$, or the coefficient of determination, is a value in the range [0, 1] and measures the proportion of variation in the data that is accounted for in the model. The equation for $R^2$ is

$$
R^2 = 1 - \frac{\sum (y_i - \widetilde{y}_i)^2}{\sum (y_i - \bar{y}_i)^2}
$$

where $\bar{y}_i$ is the mean of the actual data. Values close to 1 means the the data is strongly correlated between its variables i.e. an $R^2$ value of 0.9 means the relationship between the variables explains 90% of the variation in the data. Values close to 0 indicate little or no correlation.

We use both RMSE and $R^2$ as measures of how well our regression models fit the data when predicting our Price_Night label. The closer a value of $R^2$ to 1, the better our predictions are compared to the real data.

### Hyperparameter tuning
The next step is to then turn our attention to evaluating the model when using different sets of hyperparameters. Each model utilises different hyperparamaters. We want to assess which combination of hyperparamaters results in the lowest loss, in our case the lowest RMSE and $R^2$. Thus, we iterate over all permutations of hyperparameters, fitting the model with that specific set of hyperparameters and evaluating the models performance. The best scoring model is returned, returning the corresponding set of hyperparameters that gave us that score. sklearn has a class called GridSearchCV that implements hyperparameter tuning 

### Model evaluation
