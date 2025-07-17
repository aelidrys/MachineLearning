import numpy as np
import pandas as pd

def focal_loss(y_true, y_predict, alpha=0.25, gamma=2):

    # Focal Loss without alpha
    fLoss = -(
        y_true * np.power(1-y_predict, gamma) * np.log(y_predict) + 
        (1-y_true) * np.power(y_predict, gamma) * np.log(1-y_predict)
    )
    alpha_fLoss = -(
        alpha * y_true @ np.power(1-y_predict, gamma) * np.log(y_predict) + 
        (1-alpha) * (1-y_true) * np.power(y_predict, gamma) * np.log(1-y_predict)
    )
    return fLoss
