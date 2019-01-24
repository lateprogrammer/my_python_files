"""

Formula for Robust Scaler: (Xi - Q2)/(Q3-Q1)

This scaler removes the median and scales the data according to the quantile range

"""

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

x = np.array([[-500],
             [-100],
             [0],
             [100],
             [900]])

robust = RobustScaler()

# Scaling using Robust Scaler
x_robust = robust.fit_transform(x)

# Quartile meaures

Q1 = np.percentile(x,25)
Q2 = np.percentile(x,50)
Q3 = np.percentile(x,75)

print(Q1)
print(Q2)
print(Q3)

robust_scaler = (x-Q2)/(Q3-Q1)
print(robust_scaler)

###########################################################################################################

# For n dimensional arrays

# ROBUST SCALING

x1 = np.array([[2,2,3],
              [4,5,6],
              [7,8,9]])

robust2 = robust.fit_transform(x1)

print(robust2)

# USING QUARTILE MEASURES

"""
Q1 = (2+4)/2 = 3
Q2 = 4
Q3 = (4+7)/2 = 5.5
"""

value1 =  (2-4)/(5.5-3) # where 2 is the first value in the array
print(value1)

###########################################################################################################

# We will see the same using a pandas dataframe

import pandas as pd

df = pd.read_csv("Churn_Modelling.csv", index_col = "RowNumber")

df.head()

""" We can feature scale the EstimatedSalary Column using Robust Scaler and compare it"""

features = df.iloc[:,[11]].values

df["features_scaled"] = robust.fit_transform(features)

Q1 = np.percentile(features,25)
Q2 = np.percentile(features,50)
Q3 = np.percentile(features,75)

df["Robust"] = (features - Q2) / (Q3 - Q1)

# test_data['name','probability_predictions'].topk('probability_predictions', k=20).print_rows(20)

print(df['features_scaled'].head(), df['Robust'].head())
print("We can see that both are absolutely matching")
