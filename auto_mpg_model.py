# Auto MPG model

import pandas as pd
from math import sqrt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# read data
col = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
df = pd.read_fwf("./data/auto-mpg.data", names=col)
df.head()

# drop columns
df1= df.drop(columns=['origin', 'car_name','model_year'])
df2 = df1[df1['horsepower'] != '?']
df2['horsepower'] = df2['horsepower'].astype(float)
df2.info()

# train-test split
X=df2[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y=df2['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Get predictions
y_train_pred = lr.predict(X_train)
print(sqrt(mean_squared_error(y_train, y_train_pred)))
y_test_pred = lr.predict(X_test)
print(sqrt(mean_squared_error(y_test, y_test_pred)))

# save model
filename = 'auto_mpg_lr_model.pkl'
pickle.dump(lr, open(filename, 'wb'))