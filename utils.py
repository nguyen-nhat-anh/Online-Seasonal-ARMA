import numpy as np


def mape(true, pred):
    """
    Mean absolute percentage error
    """
    return np.mean(np.abs((true - pred) / true))


def rmse(true, pred):
    """
    root mean square error
    """
    return np.sqrt(np.mean((true - pred)**2))


def mae(true, pred):
    """
    mean absolute error
    """
    return np.mean(np.abs(true - pred))


def reshape_history(X, p, P, s):
    """
    Trích chọn và chuyển chuỗi lịch sử từ dạng [X_{t-1}, X_{t-2}, ..., X_{t-(p+P*s)}, ...]
    về dạng ma trận (p+1)x(P+1)
    0       X_{t-s}     ... X_{t-P*s}
    X_{t-1} X_{t-(1+s)} ... X_{t-(1+P*s)}
    X_{t-2} X_{t-(2+s)} ... X_{t-(2+P*s)}
    ...     ...         ... ...
    X_{t-p} X_{t-(p+s)} ... X_{t-(p+P*s)}
    """
    X_ = [0] * ((P + 1) * s)
    X_[1:p + P * s + 1] = list(X)
    X_ = np.reshape(X_, (P + 1, s)).T
    return X_[:p + 1]