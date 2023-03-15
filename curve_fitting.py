import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

class PolynomialRegression(object):
    def __init__(self, degree=2, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        #return np.abs(y - self.predict(X)).mean()
        return mean_squared_error(y, self.predict(X))

class MultiPolyFitting(object):
    def __init__(self) -> None:
        pass

    def fit(self, profile_2d):
        profile_2d_k = profile_2d.copy()
        coeffs_ls = []
        for k in range(2):
            coeffs_k, profile_2d_k, n_inlier = self.fit_1_ite(profile_2d_k, k)
            coeffs_ls.append(coeffs_k)
            #print(n_inlier, coeffs_k)
        return coeffs_ls

    def fit_1_ite(self, profile_2d, i_ite):
        if i_ite<2:
            degree = 2
            residual_threshold = 3
        else:
            degree = 2
            residual_threshold = 3
        x_vals, y_vals = profile_2d[:,0], profile_2d[:,1]
        estimator = PolynomialRegression(degree)
        ransac = RANSACRegressor(estimator,
                                min_samples=10,
                                residual_threshold=residual_threshold,
                                max_trials=10000,
                                stop_probability=0.999,
                                random_state=0)
        ransac.fit(x_vals[:,None], y_vals)
        inlier_mask = ransac.inlier_mask_
        n_inlier = inlier_mask.sum()
        if 0:
            y_hat = ransac.predict(x_vals[:,None])
            plt.figure()
            plt.plot(x_vals, y_vals, 'b.', label='input samples')
            plt.plot(x_vals[inlier_mask], y_vals[inlier_mask], 'go', label='inliers')
            plt.plot(x_vals, y_hat, 'r-', label='estimated curve')
            plt.show()

        out_mask = np.logical_not(inlier_mask)
        profile_2d_remain = profile_2d[out_mask]
        coeffs = ransac.estimator_.coeffs
        return coeffs, profile_2d_remain, n_inlier

