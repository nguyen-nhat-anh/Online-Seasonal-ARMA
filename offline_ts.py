import numpy as np
from utils import reshape_history


class SARMAPrediction:
    """
    Dự báo mô hình Seasonal ARMA(p,q)x(P,Q)_s
    """
    def __init__(self, phi, Phi, theta, Theta, s):
        """
        params:
         phi: np array shape (p,), các tham số AR
         Phi: np array shape (P,), các tham số seasonal AR
         theta: np array shape (q,), các tham số MA
         Theta: np array shape (Q,), các tham số seasonal MA
         s: int, chu kì
        """
        self.phi = phi
        self.Phi = Phi
        self.theta = theta
        self.Theta = Theta
        
        self.s = s
        self.p = len(self.phi)
        self.P = len(self.Phi)
        self.q = len(self.theta)
        self.Q = len(self.Theta)
        
        self.alpha = self._create_alpha_matrix()
        self.beta = self._create_beta_matrix()
        
    def _create_alpha_matrix(self):
        alpha = np.zeros((self.p + 1, self.P + 1))
        alpha[1:, 0] = self.phi
        alpha[0, 1:] = self.Phi
        alpha[1:, 1:] = -(self.phi[:,None] @ self.Phi[None,:])
        return alpha
    
    def _create_beta_matrix(self):
        beta = np.zeros((self.q + 1, self.Q + 1))
        beta[1:, 0] = self.theta
        beta[0, 1:] = self.Theta
        beta[1:, 1:] = self.theta[:,None] @ self.Theta[None,:]
        return beta
    
    def predict(self, X, eps):
        """
        Dự báo sử dụng quan sát và nhiễu lịch sử
        params:
         X: chuỗi quan sát lịch sử [X_{t-1}, X_{t-2}, ..., X_{t-(p+P*s)}]
         eps: chuỗi nhiễu lịch sử [e_{t-1}, e_{t-2}, ..., e_{t-(q+Q*s)}]
        returns: giá trị dự báo x_t tại thời điểm t
        """
        X_ = reshape_history(X, self.p, self.P, self.s)
        eps_ = reshape_history(eps, self.q, self.Q, self.s)
        
        x_pred = np.sum(self.alpha * X_) + np.sum(self.beta * eps_)
        return x_pred