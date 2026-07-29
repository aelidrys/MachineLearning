import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



# Visualize the precision-recall curve
def pr_curve(precision, recall, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label='Precision', color='b')
    plt.plot(thresholds, recall[:-1], label='Recall', color='r')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()


# Visualize the confusion matrix
def precisionVsRecall(precisions, recalls):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label='Precision-Recall Curve', marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.show()