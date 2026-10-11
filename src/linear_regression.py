import numpy as np
class LinearRegression:                                            
    def __init__(self,method,learning_rate,epochs):
        self.method=method 
        self.coef_=None 
        self.intercept_=None 
        self.learning_rate=learning_rate 
        self.epochs=epochs
        self.loss_history=[]
    def fit(self,X_train,y_train): 
        X_train=np.asarray(X_train,dtype='float')
        y_train=np.asarray(y_train,dtype='float').reshape(-1)

        if X_train.ndim==1:
            X_train=X_train.reshape(-1,1)

        if X_train.ndim!=2:
            raise ValueError("x_train must be 2-d")
        
        if (X_train.shape[0]!=y_train.shape[0]):
            raise ValueError("X_train and y must have same no.of samples")
        
        if (X_train.shape[0]==0 or X_train.shape[1]==0):
            raise ValueError("Training dataset cannot be empty")
        
        if self.method=="OLS":
            self.fit_ols(X_train,y_train)

        elif self.method=="gradient_descent":
            self.fit_gradient_descent(X_train,y_train)
        else:
            raise ValueError("unknown method")
        return self 

    def fit_ols(self,X_train,y_train):
        X_train=np.insert(X_train,0,1,axis=1)
        weights=np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.coef_=weights[1:]
        self.intercept_=weights[0]

    def fit_gradient_descent(self,X_train,y_train):
        X_train=np.insert(X_train,0,1,axis=1)
        betas=np.ones(X_train.shape[1])
        for i in range(self.epochs):
            y_hat=np.dot(X_train,betas)
            error=y_train-y_hat
            loss_=(np.dot(error.T,error))/X_train.shape[0]
            self.loss_history.append(loss_)
            dw=(np.dot(X_train.T,np.dot(X_train,betas))-np.dot(X_train.T,y_train))/X_train.shape[0]
            betas=betas-self.learning_rate*dw 
        self.coef_=betas[1:]
        self.intercept_=betas[0]

    def predict(self,X_test):
        if self.coef_ is None or self.intercept_ is None:
            return ValueError("Fit the model before making prediction")

        X_test=np.asarray(X_test,dtype='float')
        if X_test.ndim==1:
            X_test=X_test.reshape(-1,1)
        
        if X_test.ndim!=2:
            raise ValueError("X_test must be 2-d")

        if X_test.shape[1]!=self.coef_.shape[0]:
            raise ValueError("X_test has incorrect no.of features")

        y_pred=np.dot(X_test,self.coef_)+self.intercept_ 
        return y_pred
    
