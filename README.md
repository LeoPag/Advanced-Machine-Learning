# Advanced-Machine-Learning
This repo contains the projects for the course Advanced Machine Learning at ETH Zurich. The projects resemble Kaggle data science competitions. We were given training data and were asked to make predictions on out of sample data. Our models were evaluated both on a public leaderboard and on a private leaderboard.


## Project 0: Dummy task
Project 0 is a dummy task to help student familirize with the course framework. We were asked to read the features and to regress the mean.

## Project 1: Predict the age of a person from their MRI scan
This task is primarily concerned with regression. The original MRI features were perturbed in several ways. We needed to perform outliers detection, feature selection, and other preprocessing to achieve the best result.

The task1 was to predict a person’s age from the brain image data: a standard regression problem. The original dataset included 832 features as well as a lot of NaN values and a few outliers. A good preprocessing stage was necessary in order to have a well defined dataset that could be used in our regression model. The first step is the imputation of the dataset. At first a simple median imputation was performed. Then we decided to opt for a K-Nearest-Neighbour imputation that achieved better validation results.
As a next step, we proceeded with the feature selection process. Several algorithms were tried. At first,we eliminated the most correlated features, then we selected the most relevant features according to the Pearson Coefficient. This method ouperformed Lasso regression as a feature selection step.
A lot of outlier detection techniques were used but we decided to keep the outliers as the reduction of data almost always proved to be detrimental.
Our final predictor was an ensemble of: Multi-Layer-Perceptron, XGBoost, SupportVectorRegression,RandomForest. The R2 score of the model was 0.76 on the public leaderboard and 0.71 on the public one

