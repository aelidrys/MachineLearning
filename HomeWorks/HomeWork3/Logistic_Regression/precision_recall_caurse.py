from sklearn.metrics import precision_recall_curve, auc
from canser_classifier import logistic, sk_logistic, load_breast_cancer
import  matplotlib.pyplot as plt



def main():
    y_prop, y_gth = sk_logistic(42)
    precision, recall, thresholds = precision_recall_curve(y_gth, y_prop)
    area_under_curve = auc(recall, precision)
    plt.figure(figsize=(12,8))
    plt.plot(recall, precision, label=f"Precision Recall Curve auv = {area_under_curve:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()

