import matplotlib.pyplot as plt
import numpy as np
from model import LogisticRegression
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

threshold = 0.6

def logistic():
    X = np.array([4, 5, 6, 7, 8, 9]).reshape((-1, 1))
    y = np.array([0, 0, 0, 1, 1, 1])


    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    classes = (y_pred >= threshold).astype(int)
    print('ground', y)
    print('predct', classes)
    print(f'{np.count_nonzero(classes == y)}/{y.size}')

    # visualization assumes first 3 are class #1
    plt.plot(X[:3], y[:3], 'o', color='b', label="Cat")
    plt.plot(X[3:], y[3:], '^', color='r', label="Dog")

    plt.plot(X, y_pred, color="blue", linewidth=3)  # plot line
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot([0, 10], [threshold, threshold], color="red", linewidth=3)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()





if __name__ == '__main__':
    logistic()