import numpy as np
import pandas as pd

dataset = pd.read_csv('..\Datasets\Concrete Composite Strength\Concrete_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, n_jobs = -1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

def calc_accuracy():
    percent_sum_of_error = 0
    values = len(y_test)
    for row in range(len(y_test)):
        percent_sum_of_error += (abs((y_test[row] - y_pred[row]))/y_test[row])
    return((1-percent_sum_of_error/values)*100)

print(calc_accuracy())
    
    
