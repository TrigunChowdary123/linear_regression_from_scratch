import numpy as np
from src.linear_regression import LinearRegression
from src.metrics import RegressionMetrics
import matplotlib.pyplot as plt
#Generating synthetic data
np.random.seed(42)

X = np.random.uniform(0, 100, (100, 1))
noise = np.random.normal(0, 10, 100)

y = 2 * X.ravel() + 1 + noise

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression(method="OLS", learning_rate=0.01, epochs=1000)
lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)
y_pred=lr.predict(X_test)

mse = RegressionMetrics.mse(y_test, y_pred)
rmse = RegressionMetrics.rmse(y_test, y_pred)
r2 = RegressionMetrics.r2_score(y_test, y_pred)
print("mse:", mse)
print("rmse:", rmse)
print("r2_score:", r2)
with open("evaluation_results_ols.txt", "w") as file:
    file.write(f"MSE: {mse}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"R2 Score: {r2}\n")
plt.scatter(X, y, label="Actual data")
plt.plot(X_test, y_pred, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.legend()
output_path = "C:/Users/hp/OneDrive/Documents/linear_regression_from_stratch/regression_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("saved to:", output_path)
plt.show()
