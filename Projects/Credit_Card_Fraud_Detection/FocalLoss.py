import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # Numerically stable Sigmoid function
from sklearn.base import BaseEstimator, ClassifierMixin



class FocalLossLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression optimized with Binary Focal Loss using L-BFGS-B and exact analytical gradients.
    Fully compatible with the Scikit-Learn Estimator API.
    """
    def __init__(self, gamma=2.0, alpha=0.25, max_iter=1000, C=1.0):
        print("Initializing Focal Loss Logistic Regression")
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter
        self.C = C
        self.classes_ = np.array([0, 1])

    def _loss_and_grad(self, weights, X, y):
        w = weights[:-1]
        b = weights[-1]
        z = X @ w + b
        p = expit(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)

        # Vectorized Focal Loss calculation
        p_t = np.where(y == 1, p, 1 - p)
        alpha_t = np.where(y == 1, self.alpha, 1 - self.alpha)

        loss = np.sum(-alpha_t * ((1 - p_t) ** self.gamma) * np.log(p_t))
        l2_penalty = (0.5 / self.C) * np.sum(w ** 2)
        total_loss = loss + l2_penalty

        # Exact analytical gradients w.r.t z (dz = d(FL)/dz)
        dz_1 = self.alpha * ((1 - p) ** self.gamma) * (self.gamma * p * np.log(p) + p - 1)
        dz_0 = (1 - self.alpha) * (p ** self.gamma) * (p - self.gamma * (1 - p) * np.log(1 - p))
        dz = np.where(y == 1, dz_1, dz_0)

        # Gradients w.r.t parameters (w and b)
        dw = X.T @ dz + (1.0 / self.C) * w
        db = np.sum(dz)

        return total_loss, np.hstack([dw, db])

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_features = X.shape[1]

        init_weights = np.zeros(n_features + 1)
        
        # Optimize using L-BFGS-B (quasi-Newton method)
        res = minimize(
            fun=self._loss_and_grad,
            x0=init_weights,
            args=(X, y),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': self.max_iter}
        )

        self.w_ = res.x[:-1]
        self.b_ = res.x[-1]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.w_ + self.b_
        p1 = expit(z)
        p0 = 1 - p1
        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
