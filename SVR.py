import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

np.random.seed(42)
num_data_points = 100
days = np.arange(1, num_data_points + 1)
r = np.random.randn(num_data_points)
prices = 100 + np.cumsum(r)

data = pd.DataFrame({"days" : days,
                     "prices" : prices})


data["prices"] = data["prices"].astype(float)

X = data["Days"]  #attribute
y = data["prices"] #goal variable
