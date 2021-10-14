import numpy as np

class GaussianDiscriminant:
    def __init__(self,k=2,d=8,priors=None,shared_cov=False):
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        self.mean[0] = np.average(Xtrain[ytrain == 1], axis =0)
        self.mean[1] = np.average(Xtrain[ytrain == 2], axis =0)


        if self.shared_cov:
            # compute the class-independent covariance
            self.S = np.cov(Xtrain.T, bias = True)

        else:
            self.S[0] = np.cov(Xtrain[ytrain == 1].T, ddof=0)
            self.S[1] = np.cov(Xtrain[ytrain == 2].T, ddof=0)

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            g = [0, 0] # list to hold descriminant values
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                if self.shared_cov:
                    g[c] = -0.5 * (Xtest[i] - self.mean[c]).T @ np.linalg.inv(self.S) @ (Xtest[i] - self.mean[c]) + np.log(self.p[c])
                else:
                    # term1 = (-0.5 * (np.log(np.linalg.det(self.S[c]))))
                    # term2 = (Xtest[i].T @ np.linalg.inv(self.S[c])) @ Xtest[i]
                    # term3 = 2 * Xtest[i].T @ np.linalg.inv(self.S[c]) @ self.mean[c]
                    # term4 = self.mean[c].T @ np.linalg.inv(self.S[c]) @ self.mean[c]
                    # g[c] = term1 -(0.5 * (term2 - term3 + term4)) + np.log(self.p[c])
                    term1 = (-0.5 * (np.log(np.linalg.det(self.S[c]))))
                    term2 = 0.5 * ((Xtest[i] - self.mean[c]).T @ np.linalg.inv(self.S[c]) @ (Xtest[i] - self.mean[c]))
                    g[c] = term1 -0.5 * term2 + np.log(self.p[c])

            # determine the predicted class based on the values of discriminant function
            if(g[0] < g[1]):
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Diagonal:
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d
    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        self.mean[0] = np.average(Xtrain[ytrain == 1], axis =0)
        self.mean[1] = np.average(Xtrain[ytrain == 2], axis =0)

        # compute the class-independent covariance
        for i in range(0, self.d):
            self.S[i] = np.var(Xtrain[:,i])

    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            # calculate the value of discriminant function for each class
            g = [0,0]
            for c in np.arange(self.k):
                term1 = -0.5 * np.sum(np.multiply(self.S**(-1), ((Xtest[i] - self.mean[c])**2)))
                g[c] = term1 + np.log(self.p[c])

            # determine the predicted class based on the values of discriminant function
            if(g[0] < g[1]):
                predicted_class[i] = 2
                
        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
