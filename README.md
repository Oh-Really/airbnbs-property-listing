### Modelling Airbnb'a Property dataset

In this project, we aim to use a dataset of Airbnb properties along with accompanying images to build, train, tune and evalueate models on that dataset.

The project is split into a few parts, namely;
- **Data cleaning and handling**. Here we evaluate the input airbnb data and clean it by removing certain rows with null values, cleaning the descriptions column into a readable string, as well as selecting only numerical parts of the whole dataframe.
- **Creating a Baseline model**. During this step, we create a Linear Regression model and evaluate its performance on our validation set using metrics such as RMSE and $R^2$.