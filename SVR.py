import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
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


data["days"] = data["days"].astype(float)

X = data[["days"]]  #attribute
y = data["prices"] #goal variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVR(kernel='linear')
model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)




new_days = np.arange(num_data_points + 1, num_data_points + 11)
new_days_df = pd.DataFrame({"days": new_days})
new_days_scaled = scaler.transform(new_days_df)
new_price = model.predict(new_days_scaled)


plt.figure(figsize=(5, 5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label= 'Testing Data')
plt.scatter(new_days, new_price, color='pink', label= 'Predicting Data')
plt.plot(X_train, y_pred_train, color='red', label='Training Prediction')
plt.plot(X_test, y_pred_test, color='orange', label='Testing Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()