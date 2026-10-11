import numpy as np
class RegressionMetrics:
    @staticmethod
    def mse(y_true,y_pred):
      residual=y_true-y_pred 
      return np.sum((residual)**2)/y_true.shape[0]

    @staticmethod 
    def rmse(y_true,y_pred):
        residual=y_true-y_pred 
        return np.sqrt(np.sum((residual)**2)/y_true.shape[0])

    @staticmethod
    def r2_score(y_true,y_pred):
       y_true_mean=np.mean(y_true)
       return 1-(np.sum((y_true-y_pred)**2))/(np.sum((y_true-y_true_mean)**2))

   