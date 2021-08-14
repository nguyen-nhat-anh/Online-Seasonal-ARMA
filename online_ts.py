import numpy as np
from collections import deque
from utils import reshape_history


class SARMA_OGD:
    """
    This class implements Seasonal ARMA - Online Gradient Descent algorithm
    """
    def __init__(self, p, P, s, X_max):
        """
        params:
         p: int, bậc tự hồi qui trend
         P: int, bậc tự hồi qui seasonal
         s: int, chu kì mùa
         X_max: float, chặn trên trị tuyệt đối của chuỗi thời gian
        """
        self.p = p
        self.P = P
        self.s = s
        
        self.c = 1
        self.X_max = X_max
        self.t = 1
        
        # chuỗi lịch sử [X_{t-1}, X_{t-2}, ..., X_{t-(p+Ps)}]
        self.X = np.zeros(p + P * s)
        self.X = deque(self.X, maxlen=p + P * s)
        
        # hệ số mô hình
        self.gamma = np.random.uniform(-self.c, self.c, (p + 1, P + 1))
        self.gamma[0, 0] = 0
    
    def fit_one_step(self, x):
        """
        Run one iteration of the algorithm
        
        params:
         x: float, observation value at t (X_t)
        
        returns:
         x_pred: float, prediction value at t (Xpred_t)
         loss: float, squared loss (x_pred - x)^2
        """
        # predict
        X_ = reshape_history(self.X, self.p, self.P, self.s)
        x_pred = np.sum(self.gamma * X_)
        
        # compute loss
        loss = (x_pred - x)**2
        
        # update parameters
        grad = 2 * (x_pred - x) * X_
        self.gamma -= grad / (self.X_max**2 * np.sqrt(self.t))
        self.gamma = np.clip(self.gamma, -self.c, self.c)
        
        # update history
        self.X.appendleft(x)
        
        self.t += 1
        
        return x_pred, loss