import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc



# Visualize the precision-recall curve
def pr_curve(precision, recall, thresholds):
    plt.figure(figsize=(8, 6))
    auc_score = auc(recall, precision)
    plt.plot(thresholds, precision[:-1], label='Precision', color='b')
    plt.plot(thresholds, recall[:-1], label='Recall', color='r')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision-Recall Curve (AUC = {auc_score:.2f})')
    plt.legend()
    plt.grid()
    plt.show()


# Visualize the confusion matrix
def precisionVsRecall(precisions, recalls):
    plt.figure(figsize=(8, 6))
    auc_score = auc(recalls, precisions)
    plt.plot(recalls, precisions, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})', marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.show()