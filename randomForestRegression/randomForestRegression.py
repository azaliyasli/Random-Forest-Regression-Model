import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, -1].values

#Training
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

#Prediction
y_pred = regressor.predict([[6.5]])
print(y_pred)

#Visualising
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Position Salaries')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()