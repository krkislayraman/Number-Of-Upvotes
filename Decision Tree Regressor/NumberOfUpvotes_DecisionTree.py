# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 19:48:28 2019

@author: RAMAN
Number of upvotes - DT - Regressor
"""

import os
import numpy as np
import pandas as pd
import pydotplus
import math
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

os.chdir("C:/Users/RAMAN/Documents/R/Analytics Vidya/Number of Upvotes")
os.getcwd()

# Read the data
train_data = pd.read_csv('train_NoU.csv')
test_data = pd.read_csv('test_NoU.csv')

describe = train_data.describe(include = 'all')

train_data.nunique()

train_data.dtypes

# Create dummy variables
categ_vars = train_data.loc[:, train_data.dtypes == object].columns
dummy_df = pd.get_dummies(train_data[categ_vars], dtype = int)
dummy_df.dtypes

full_data = pd.concat([train_data, dummy_df], axis=1)
full_data.shape
full_data.dtypes
fulldata = full_data.drop(['Tag', 'ID', 'Username'], axis=1).copy()
fulldata.info()
fulldata.shape

# Seperating dependent and independent variables from dataset
Y = fulldata['Upvotes']
X = fulldata.drop(['Upvotes'], axis = 1)

# Confirm the changes
X.shape
Y.shape

# Random Sampling into Train_X, Test_X, Test_Y, Train_Y
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, test_size = 0.3, random_state = 100)

# Decision Tree Model
M1 = DecisionTreeRegressor(criterion="mse", random_state = 100)
M1_Model = M1.fit(Train_X, Train_Y)


# Scores from sklearn.metrics
model_score_train = M1_Model.score(Train_X,Train_Y) # 0.9999999980615071

Test_Pred = M1_Model.predict(Test_X)

model_score_test = M1_Model.score(Test_X,Test_Y) # 0.842298226033708

M1_Model.feature_importances_

Var_Importance_Df = pd.concat([pd.DataFrame(Train_X.columns).rename(columns = {0:'Colnames'}), pd.DataFrame(M1_Model.feature_importances_)], axis = 1)
Var_Importance_Df

# Score
r2_score(Test_Y, Test_Pred) # 0.842298226033708
math.sqrt(mean_squared_error(Test_Y, Test_Pred)) # 1376.878189283463

################################## Test on actual test set ####################################33
# Create dummy variables
categ_vars_test = test_data.loc[:, train_data.dtypes == object].columns
dummy_df_test = pd.get_dummies(test_data[categ_vars], dtype = int)
dummy_df_test.dtypes

full_data_test = pd.concat([test_data, dummy_df_test], axis=1)
full_data_test.shape
full_data_test.dtypes
fulldata_test = full_data_test.drop(['Tag', 'ID', 'Username'], axis=1).copy()

# predict

raw_Test_Pred = M1_Model.predict(fulldata_test)

from pandas import DataFrame

submission = DataFrame(raw_Test_Pred)
submission.to_csv('submission.csv')

# 1476.2800989323948.

# Plot
# Create dot data
dot_data = export_graphviz(M1_Model, out_file=None, feature_names = Train_X.columns, max_depth = 6, filled = True)

# Draw Graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph (on console) # takes a bot of time
#Image(graph.create_png())

# Write to a file
graph.write_pdf("DT_Plot_Number_Of_Upvotes.pdf")
os.getcwd()

export_graphviz(M1_Model, out_file ='tree.dot', 
               feature_names =Train_X.columns)  
