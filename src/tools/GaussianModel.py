import numpy as np
from scipy.stats import multivariate_normal

class GMM(object):

    weights = None
    means = None
    covars = None
    k=None
    iterations =100
    convergence_th=1e-3
    ric=None

    def __init__(self, n_components=1, tol=1e-3, max_iter=100):
        """
        A Gaussian mixture model trained via the expectation maximization
        algorithm.

        Parameters
        ----------

        n_components: The number of mixture components.
        tol: The convergence threshold
        """
        self.k=n_components
        self.iterations=max_iter
        self.convergence_th=tol


    def initialize_params(self, X, kmeans=False):
        """
        Initialize the starting GMM parameters.

        Parameters
        ----------
        X : A collection of `N` training data points, each with dimension `d`.
        kmeans: A boolean flag for determining if to initialize the params with kmeans or randomly.
        """

        self.means = np.random.rand(self.k,X.shape[1])
        self.covars = np.zeros((self.k, X.shape[1], X.shape[1]))
        for k in range(self.k):
            self.covars[k] = np.eye(X.shape[1])
        self.weights=[1/self.k]*self.k
        self.ric = np.zeros((X.shape[0], self.k))

    def formula(self,x,u,sigma,d):
        first_term = 1/(((2*np.pi)**(d/2))*np.sqrt(sigma))
        exp = -0.5*(np.dot(np.dot((x-u).T,np.linalg.inv(sigma)),(x-u)))
        second_term = np.exp(exp)
        return first_term*second_term

    def E_step(self, X):
        """
        Find the Expectation of the log-likelihood evaluated using the current estimate for the parameters

        Parameters
        ----------
        X : A collection of `N` data points, each with dimension `d`.
        """
        n=X.shape[0]
        d = X.shape[1]

        nan_mask = np.isnan(self.covars)
        inf_mask = np.isinf(self.covars)
        self.covars[nan_mask] = 1e-6
        self.covars[inf_mask] = 1e-6

        for c in range(self.k):
            self.ric[:,c]= self.weights[c]*multivariate_normal.pdf(X,mean=self.means[c],cov=self.covars[c],allow_singular=True)
        for i in range(n):
            self.ric[i] = self.ric[i]/np.sum(self.ric[i])
        return

    def M_step(self, X):
        """
        Updates parameters maximizing the expected log-likelihood found on the E step.

        Parameters
        ----------
        X : A collection of `N` data points, each with dimension `d`.
        """
        n=X.shape[0]
        d=X.shape[1]

        for c in range(self.k):
            self.weights[c]=np.sum(self.ric[:,c])/n

        for c in range(self.k):
            clusterSum = np.sum(self.ric[:, c])
            self.means[c] = np.sum(X * self.ric[:, c][:, np.newaxis], axis=0) / clusterSum
            diff = X - self.means[c]

            self.covars[c] = (np.dot(self.ric[:, c] * diff.T, diff)/ clusterSum)
            self.weights[c] = clusterSum / n



    def fit(self, X, y=None):
        """
        Fit the parameters of the GMM on some training data.

        Parameters
        ----------
        X : A collection of `N` training data points, each with dimension `d`.
        y: not used
        """
        self.initialize_params(X)
        for i in range(self.iterations):
            self.E_step(X)
            self.M_step(X)

    def predict(self, X):
        """
        Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : A collection of `M` data points, each with dimension `d`.

        Returns
        -------
        Predicted labels.
        """
        N = X.shape[0]
        gamma = np.zeros((N, self.k))
        for k in range(self.k):
            pdf = multivariate_normal(mean=self.means[k], cov=self.covars[k])
            gamma[:, k] = self.weights[k] * pdf.pdf(X)
        return np.argmax(gamma, axis=1)