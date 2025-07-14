import numpy as np
import pandas as pd

def focal_loss(y_true, y_predict, alpha=0.25, gamma=2):
    # y_true*log(y_predict) - (1-y_true)*log(1-y_predict) Cross Entropy
    # y_true*(1-y_predict)^gamma*log(y_predict) - (1-y_predict)*y_predict^gamma*log(1-y_predict) Focal Loss
    pass
