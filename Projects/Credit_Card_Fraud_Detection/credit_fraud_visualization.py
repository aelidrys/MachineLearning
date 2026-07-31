import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc



# Visualize the precision-recall curve
def pr_curve(precision, recall, thresholds):
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    auc_score = auc(recall, precision)
    axes[0].plot(thresholds, precision[:-1], label='Precision', color='b')
    axes[0].plot(thresholds, recall[:-1], label='Recall', color='r')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Precision-Recall Curve')
    axes[0].legend()
    axes[0].grid()

    auc_score = auc(recall, precision)
    axes[1].plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})', marker='.')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve  (AUC = {auc_score:.2f})')
    axes[1].grid()
    plt.show()


