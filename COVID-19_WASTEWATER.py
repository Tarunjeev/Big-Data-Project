#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import  datetime
import matplotlib.pyplot as plt
from sklearn.impute import MissingIndicator
from datetime import datetime, timedelta
get_ipython().system(' pip install openpyxl')
get_ipython().system(' pip install xgboost')
get_ipython().system(' pip install knnmv')
get_ipython().system(' pip install lightgbm')
import xgboost as xg
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from knnmv.impute import KNNMVImputer
from sklearn.metrics import mean_squared_error


# In[ ]:


import warnings warnings.filterwarnings("ignore")
# Reading data
pred_data = pd.read_csv('../input/stat440-21-module1/predictions.csv')
cases_data = pd.read_excel(open('../input/stat440-21-module1/BC COVID CASES.xlsx', 'rb'), she waste_water_data = pd.read_excel(open('../input/stat440-21-module1/BC COVID CASES.xlsx', 'rb'


# In[ ]:


cases_data.info()
# Heatmap
corrmat = cases_data.corr()
g = sns.heatmap(cases_data.corr())
# most correlated features
top_corr_features = corrmat.index[abs(corrmat['New cases'])>0.5]
g = sns.heatmap(cases_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# To see the missing percentage of values =
alldatana =  (cases data isnull() sum() / len(cases data)) * 100
alldatana = (casesdataisnull()sum()/len(casesdata)) 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False) 
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data


# In[ ]:


cases_data.agg(['skew']).transpose()


# In[ ]:


# preparing waste water dataset for combining
df = waste_water_data.copy()
df = df.groupby(['Date', 'Plant']).size().unstack(fill_value=0).reset_index().rename_axis(Non tmp = df.Date
fg = pd.DataFrame(tmp,columns = ['Date'])
for x in waste_water_data.Plant.unique():
    Annacis_df = waste_water_data.copy()
    Annacis_df = Annacis_df[Annacis_df['Plant'] == x].rename(columns = {"Count": x + ".Count" })                
    fg = fg.join(Annacis_df.set_index('Date'), on='Date').reset_index()
    fg.drop('index', axis=1, inplace=True)
                                                                                          
for x in fg.iloc[:,fg.columns != "Date"]:
    plt.plot(fg.Date,fg[x], label = x) 
    plt.xticks(rotation=45) plt.xlabel('Date')
    plt.ylabel('RNA Count')
    plt.legend(loc="upper left")
    
plt.show()


# In[ ]:


# Attempt to check skewness of waste water data # fg.agg(['skew', 'kurtosis']).transpose()
# # skewed data
# skew1 = fg.copy()
# #skew = skew.loc[:, cases_data.columns != 'New cases']
# skew1 = skew1.loc[:, skew1.columns != 'Date']
# for feature in skew1.columns:
# skew1[feature] = np.log1p(skew1[feature])
# #skew['New cases'] = cases_data['New cases']
# skew1['Date'] = fg['Date']
# skew1.agg(['skew', 'kurtosis']).transpose()
# Dateframe created from combining cases and wastewater pages of the BC COVID CASES excel fil
combined_covid_date = cases_data.join(fg.set_index('Date'), on='Date').reset_index() 
combined_covid_date.drop('index', axis=1, inplace=True)
indicator = MissingIndicator()
for x in combined covid date.iloc[:,combined covid date.columns!='Date'].columns:
    combined_covid_date[x + '.indicator'] = indicator.fit_transform(combined_covid_date[['Dat combined_covid_date[['Annacis Island.Count', 'Iona Island.Count', 'Lions Gate.Count',
    'Lulu Island.Count', 'Northwest Langley.Count']] = combined_covid_date[['Annacis Islan
    'Lulu Island.Count', 'Northwest Langley.Count']].fillna(0) 
combined_covid_date.head()


# In[ ]:


# Retrieving the unique target dates from the prediction.csv file 
target_dates = pred_data['Date:Delay'].str.split(pat=':', expand=True)[0] 
day_diff = pred_data['Date:Delay'].str.split(pat=':', expand=True)[1] 
#day_diff = day_diff.astype(int)
print("Unique target dates:", target_dates.unique().size)


# In[ ]:



# Code to add indicator variables for rows with nan values 
indicator = MissingIndicator()
for x in cases_data.iloc[:,cases_data.columns!='Date'].columns:
    cases_data[x + '.indicator'] = indicator.fit_transform(cases_data[['Date',x]].values)*1


# In[ ]:


# Code to create a real data subset for intermal rmse testing
# test_data = list()
# temp_impute = target_dates
# for i in range(target_dates.size):
#val = (test_data[test_data['Date'] == temp_impute.iloc[i]]['Cumulative cases'].iloc[0 test_data.append(cases_data.iloc[cases_data[cases_data['Date'] == temp_impute[i]].index


# In[ ]:


# # #print(test_data)
# pd.DataFrame(list(zip(pred_data['Date:Delay'], test_data)),columns =['Date:Delay', 'Count']


# In[ ]:



# correcting skewness for cases_data
skew = cases_data.copy()
skew = skew.loc[:, cases_data.columns != 'New cases'] 
skew = skew.loc[:, skew.columns != 'Date']
for feature in skew.columns:
    skew[feature] = np.log1p(skew[feature]) 
skew['New cases'] = cases_data['New cases'] 
skew['Date'] = cases_data['Date']


# In[ ]:


# Function to retrieve the data for dates before the target date. 
def training_set_creater(target_date, days_to_subtract, data):
    train_data = data[target_date - timedelta(days=days_to_subtract) >= data['Date']] 
    return train_data
def tarun_imputer(data,col_change, col_used, days):
    # new cases imputation using formula
    # For example:
    # New cases[t] = Cumulative cases[t + 1] - Cumulative cases[t - 1] # beacuse Cumulative cases is the total number of cases
    iu = data.copy()
    temp_impute = data[data[col_change].isnull()]['Date']
    
for i in range(temp_impute.size):
    val = (iu[iu['Date'] == (temp_impute.iloc[i] + timedelta(days=days))][col_used].iloc[ data.at[iu[iu['Date'] == temp_impute.iloc[i]].index , col_change ] = val
    val = (data[data['Date'] == temp_impute.iloc[i]][col_change].iloc[0] + iu[iu['Date'] 
    data.at[iu[iu['Date'] == temp_impute.iloc[i]].index , col_used ] = val
    return data


# In[ ]:



# Function to predict the number of new cases using the test and train data. 
def predict_due_dates(target_date, days_to_subtract, data,sign):
    target_date = datetime.fromisoformat(target_date)
    train_data = training_set_creater(target_date, int(days_to_subtract), data) print(train_data["Date"])
    #################################################
    # Conversion of Date for usage in the model#
    train_data.loc[:, 'Date_year'] = train_data['Date'].dt.year
    train_data.loc[:, 'Date_month'] = train_data['Date'].dt.month
    train_data.loc[:, 'Date_week'] = train_data['Date'].dt.isocalendar().week.astype('int64') train_data.loc[:, 'Date_day'] = train_data['Date'].dt.day
    train_data.loc[:, 'Date_dayofweek'] = train_data['Date'].dt.dayofweek
    train_data = train_data.loc[:, train_data.columns != 'Date'] ######################################################################## lakshay_imputer(cases_data.iloc[0:34,:],"New cases",'Cumulative cases',1) lakshay_imputer(cases_data.iloc[0:34,:],"Deaths",'Cumulative deaths',1) ######################################################################
    # using KNN based imputation stratergy (data auto normalized)
    knnmv_imp = KNNMVImputer(strategy="mean", k=3, l=0.25)
    sample = train_data.copy()
    sample = sample.drop(['New cases','Date_year', 'Date_month', 'Date_week', 'Date_day', 'Da sample = sample.values
    ab = knnmv_imp.fit_transform(sample)
    sample1 = train_data.copy()
    sample1 = sample1.drop(['New cases','Date_year', 'Date_month', 'Date_week', 'Date_day', ' df = pd.DataFrame(ab,columns=sample1.columns)
    df[['New cases','Date_year', 'Date_month', 'Date_week', 'Date_day', 'Date_dayofweek']] = ##########################################
    # Creating Training and testing sets
    X = df.loc[:, df.columns != 'New cases']
    X_= X.iloc[-1,:]
    X = X.iloc[:-int(days_to_subtract),:]
    y = df['New cases'].shift(periods= -int(days_to_subtract)) # -1 * days_to_subtract
    y = y.iloc[:-int(days_to_subtract)]
    # Row of data used for the final prediction
    # To see if the date being passed is correct or not
    test_data = X_
    print(X[['Date_year', 'Date_month', 'Date_week', 'Date_day', 'Date_dayofweek']]) print(y)
    print(test_data)
    test_data = X.iloc[-1,:].values.reshape(1,test_data.shape[0]) # -1 * days_to_subtract
# To get the final predicted result result = predictor(X,y,test_data,sign)
    return result


# In[ ]:


def predictor(X,y,test_data,sign):
# create an xgboost regression model
    regr = xg.XGBRegressor(colsample_bytree=0.5, gamma=0.0468, learning_rate=0.05, max_depth=7,
    min_child_weight=0.5, n_estimators=2200, reg_alpha=0.8, reg_lambda=0.7, subsample=0.5213,
    random_state =7, nthread = -1)
    regr.fit(X, y)
    res = regr.predict(test_data)[0]
    return res


# In[ ]:


input_data = combined_covid_date
pred_list2 = list()
for i in range(target_dates.size): #target_dates.size
    #print(i)
    pred_list2.append(predict_due_dates(target_dates[i], day_diff[i],input_data,1 )) 
pd.DataFrame(list(zip(pred_data['Date:Delay'], pred_list2)),columns =['Date:Delay', 'Count'])
# skewed data for combined dataset
skew1 = combined_covid_date.copy()
#skew = skew.loc[:, cases_data.columns != 'New cases'] skew1 = skew1.loc[:, skew1.columns != 'Date']
for feature in skew1.columns:
    skew1[feature] = np.log1p(skew1[feature])
#skew['New cases'] = cases_data['New cases'] 
skew1['Date'] = fg['Date']


# In[ ]:


# # hyper parameter tuning
# def para_tuning(target_date, days_to_subtract, data,sign):
# target_date = datetime.fromisoformat(target_date)
# train_data = training_set_creater(target_date, int(days_to_subtract), data)
#     #################################################
# #train_data = train_data.iloc[:-int(14),:]
#     #################################################
#     # Conversion of Date for usage in the model
#     ############### Temporary decision to remove date column ###################
# train_data.loc[:, 'Date_year'] = train_data['Date'].dt.year
# train_data.loc[:, 'Date_month'] = train_data['Date'].dt.month
# train_data.loc[:, 'Date_week'] = train_data['Date'].dt.isocalendar().week.astype('int64
# train_data.loc[:, 'Date_day'] = train_data['Date'].dt.day
# train_data.loc[:, 'Date_dayofweek'] = train_data['Date'].dt.dayofweek
# train_data = train_data.loc[:, train_data.columns != 'Date']
#     ########################################################################
# #knnmv_imp = KNNMVImputer(strategy="median", k=5, l=0.25)
# sample = train_data.copy()
# sample = sample.drop(['New cases'], axis = 1)
# sample = sample.values
#     #print((sample))
# ab = knnmv_imp.fit_transform(sample)
# sample1 = train_data.copy()
# sample1 = sample1.drop(['New cases'], axis = 1)
# df = pd.DataFrame(ab,columns=sample1.columns)
# # filled na's in new cases column with 0 , not sure
# df['New cases'] = train_data['New cases'].fillna(0)
# #print(df.head())
#     ##########################################
# X = df.loc[:, df.columns != 'New cases']
# X = X.iloc[:-int(days_to_subtract),:]
# y = df['New cases'].shift(periods= -int(days_to_subtract)) # -1 * days_to_subtract
# y = y.iloc[:-int(days_to_subtract)]
#     # Row of data used for the final prediction
# test_data = X.iloc[-1,:]
# test_data = test_data.values.reshape(1,17) # -1 * days_to_subtract
# from sklearn.model_selection import GridSearchCV
#     # To get the final predicted result
# params = { ''''
#objective:[reg:squarederror]
#              objective :[ reg:squarederror ],
# # # #
# # # # # # # #
# #
# 'max_depth': [3,6,7], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 500, 1000], 'colsample_bytree': [0.3, 0.7 ,0.8]}
# xgbr = xg.XGBRegressor(seed = 20) clf = GridSearchCV(estimator=xgbr,
# param_grid=params, scoring='neg_mean_squared_error', verbose=1)
# clf.fit(X, y)
# print("Best parameters:", clf.best_params_) print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
# return clf.best_params_


# ### BONUS QUESTION

# In[ ]:


# rainfall_data = pd.read_csv('../input/rainstatdata/weatherstats_vancouver_daily_stat.csv') # # rainfall_data.head()
# # converting date to datetime format
# # loaded the rain data and converted date into datetime object
# rainfall_data['date'] = pd.to_datetime(rainfall_data['date']) # rainfall_data.columns
# #filtering two columns to rain_data
# # Filtered the columns from the rain data
# # And renamed the date to Date
# rain_data = rainfall_data[['date','rain']]
# rain_data.rename(columns = {'date' : 'Date'}, inplace = True)
# rain_data.head()
# # #joining wastewater and rain on date
# # #combined_rain_waste = pd.merge(waste_water_data, rain_data, on='Date', how='inner')
# # combining the waste water data with the rain data and dropping plant and index column
# combined_rain_waste = waste_water_data.join(rain_data.set_index('Date'), on='Date').reset_i # combined_rain_waste.drop('index', axis=1, inplace=True)
# combined_rain_waste.drop('Plant', axis=1, inplace=True)
# combined_rain_waste
# # #combined_rain_waste[combined_rain_waste.rain == 0]
# # # df2 = combined_rain_waste[combined_rain_waste.rain > 0.0]
# # # df2.head()
# # #combined_rain_waste.head()
# # making new dataframe by taking Date and New cases column from cases data and
# # combining with the dataframe combined_rain_waste
# # i.e from combining wastewater and rain data
# # now spliting into two dataframe, one with
# bonus_data_set = cases_data.copy()
# bonus_data_set = bonus_data_set[['Date','New cases']]
# #Annacis_df = Annacis_df[Annacis_df['Plant'] == x].rename(columns = {"Count": x + ".Count"}
  
# bonus_data_set = bonus_data_set.join(combined_rain_waste.set_index('Date'), on='Date').rese
# bonus_data_set.drop('index', axis=1, inplace=True)
# bonus_data_set = bonus_data_set.dropna().reset_index()
# bonus_data_set.drop('index', axis=1, inplace=True)
# print(bonus_data_set.shape)
# bonus_data_set_no_rain = bonus_data_set[ bonus_data_set['rain'] == 0.0 ]
# bonus_data_set_rain = bonus_data_set[ bonus_data_set['rain'] != 0.0 ]
# bonus_data_set_no_rain.head()
# bonus_data_set_rain.head()
# bonus_data_set_rain["ratio"] = bonus_data_set_rain["Count"] / bonus_data_set_rain["New case
# bonus_data_set_no_rain["ratio"] = bonus_data_set_no_rain["Count"] / bonus_data_set_no_rain[
# bonus_data_set_rain["ratio"].hist()
# bonus_data_set_rain["ratio"]
# bonus_data_set_no_rain["ratio"].hist() # from scipy import stats
# t_value,p_value=stats.ttest_ind(bonus_data_set_rain["ratio"],bonus_data_set_no_rain["ratio"
# print('Test statistic is %f'%float("{:.6f}".format(t_value)))
# print('p-value for two tailed test is %f'%p_value)
# alpha = 0.05
# if p_value<=alpha:
# print('We reject the null hypothesis H0.') # else:
# print('We do not reject the null hypothesis H0')

