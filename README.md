# Big-Data-Project
COVID -19 Wasterwater surveillance kaggle competiton
1 INTRODUCTION
This module studies the case counts of the current COVID-19 pandemic using various machine learning method- ologies. 
Data about the COVID-19 cases in B.C. and wastewaster surveillance data is obtained from the following Kaggle competition. 
The goal of this module is to use 550 x 14 case data points and 258 x 3 wasterwater data points to predict 2060 case counts
for intervals of 1, 3, 5 or 7 days before the specified date.
Before conducting any prediction modeling, pre-processing of the data was done to handle missing-data and irregular data points.
This is an important step in the project and it likely had a substantial impact on our RMSE scores. 
After prepossessing the data, we were able to use machine learning techniques such as Multiple Linear Regression, 
Ada-Boost Regression, Ensemble Regression, and XG Boost Regression to choose the method with the lowest RMSE score. 
The method that provided our team with the lowest RMSE score, equal to 103.76218, is XG Boost.

PREPROCESSING
In order to combine both data sets and answer the challenge question, we first transformed the Plant Column 
in waste_water dataset into 5 different columns containing the RNA count in respect to that station to make it 
compatible with the dates in the cases_datset (solving the 700+ rows issue).
To establish a backtesting framework, we subset the entire dataset by rows that were before the target date. 
Then, we transformed the data column into 5 additional columns of ’Date year’, ’Date month’, ’Date weekday’, ’Date day’, 
and ’Date day of the week’ to use dates in modeling.
Also, we corrected the skewness of the data by using a log transformation. Lastly, NaN values were handled by an K-means based 
imputation method [Belloni, 2018]. This method normalized the data.

Multiple Linear Regression
Firstly, based on the lectures we tried various forms of Linear regression. Starting from regressing New cases with row numbers.
This served as a base line for all further approaches. Next, we began Linear regression on basis of multiple variables. 
We observed a decrease in score from the baseline RMSE score obtained from the aforementioned method on the Kaggle Leaderboard.
Within the testing of our linear regression model, we made sure that the distribution and the variance of the dependent variable was 
normalized and constant for all values of the independent variable.
During our exploration process, we tried to determine a mathematical relationship among several random variables which further helped
us examine how multiple independent variables were related to our dependent variable ’New cases’. Once all of the independent factors
were determined, the information on the multiple variables was used to create prediction of the dependent variable on the level 
of effect that they have on the out- come variable. Notably, the information on the multiple variables can be used to create an
accurate prediction based on the impact they have on the outcome variable.

AdaBoost regression
The second approach was AdaBoost regression. Transitioning from linear regression to Adaboost allows us to secure non-linear relationships, 
which produces a better prediction accuracy.
Weak models are inserted sequentially and trained using the weighted training data. 
The main reason to shift to a tree based model like AdaBoost is that it allows us to handle outliers and unexpected changes in the data points.
The RMSE score received from non hyper parameterized model was 120, which is significantly worse than the baseline.
But still we believed a tree based model would be the best answer after plotting the RNA counts in the wastewater data set.

Ensemble regression
The third approach was Ensemble methods. Ensemble learning combines various base algorithms to construct one optimized predictive algorithm.
We took advantage of the Parallel Learning technique of ensembles. In essence, generate base models parallel to each other and take an advantage
of the independence between models by averaging the known mistakes.Subsequently, we used the three models GBoost, Light GBoost, and XG Boost.
We trained the three mod- els separately and averaged the mean. The initial RMSE scores obtained from the mean of all three models surpassed the threshold.
However, we observed that the score of XG Boost alone surpassed the mean value of all three models.
Furthermore, GBoost and Light GBoost don’t have the innate NaN value imputation capability. 
We observed a similar trend even after a small round of Cross validation with n estimators.
The RMSE score obtained here was adequate to some extent but it is unacceptable as the final score.

XG boost regression
XGBoost is a decision tree based ensemble algorithm that uses a gradient boosting framework. It utilizes decision trees, bagging and boosting, 
which is the process of minimizing the errors of the previous models while boosting the influence of better performing models. For this module, 
we trained a XGBoost model using a RMSE loss function. It outperformed all other attempted models. 
Consequently, we began hyperparamter tuning it on max_depth, learning_rate, n_estimators, colsample_bytree using a Grid search CV.

Results
Nevertheless, various methods were considered and implemented to minimize the RMSE score using the B.C. 
COVID-19 cases information and wastewater dataset. We explored various pre-processing methodologies together as a team and our best model 
proved to be a XG Boost algorithm, providing us an RMSE of 103.76218 on Kaggle.

Lessons Learned
Through the course of this project, we learned about data prepossessing and fully comprehend it’s significance on the RMSE score.
We got the opportunity to work with time-series data for the first time in a real world setting and to research more about prediction
and data imputation methods unique to time series problems. Moreover, we also learned about the effects of collinearity and skewness in data, 
and how it affects the RMSE score acquired from a regression model.
