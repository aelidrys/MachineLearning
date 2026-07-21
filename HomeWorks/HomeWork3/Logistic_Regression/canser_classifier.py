import matplotlib.pyplot as plt
import numpy as np
from model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression as Sk_LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')


def data_preparation(add_intercept=True, random_state=42):
    data = load_breast_cancer()
    X, t = data.data, data.target.reshape((-1, 1))
    X = MinMaxScaler().fit_transform(X)
    
    if add_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.3, shuffle=True,
        stratify=t ,random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def logistic(random_state):
    print("Local Model\n")
    X_train, X_test, y_train, y_test = data_preparation(random_state=random_state)
    model = LogisticRegression()
    model.fit(X_train, y_train, _step_size=1, _precision=1e-7, _max_iter=100000)

    # training
    print('Training:')
    y_propa = model.predict(X_train)
    y_pred = (y_propa >= 0.5).astype(int)
    print(f'\t{np.count_nonzero(y_pred == y_train)}/{y_train.size}')
    print(f'\taccuracy: {accuracy_score(y_train, y_pred):.4f}')

    # testing
    print('Testing:')
    y_test_propa = model.predict(X_test)
    y_test_pred = (y_test_propa >= 0.5).astype(int)
    print(f'\t{np.count_nonzero(y_test_pred == y_test)}/{y_test.size}')
    print(f'\taccuracy: {accuracy_score(y_test, y_test_pred):.4f}')
    report_train = classification_report(y_train, y_pred)
    report_test = classification_report(y_test, y_test_pred)
    print('Training Report:\n%s' % report_train)
    print('Testing Report:\n%s' % report_test)
    return y_train, y_propa

def sk_logistic(random_state):
    print("Sklear Model\n")
    X_train, X_test, y_train, y_test = data_preparation(random_state=random_state)
    model = Sk_LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_propa = model.predict_proba(X_train)[:, 1]
    y_pred_test_prop = model.predict_proba(X_test)[:, 1]

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print('Training:')
    print(f'\taccuracy: {accuracy_train:.4f}')
    print('Testing:')
    print(f'\taccuracy: {accuracy_test:.4f}')

    # report_train = classification_report(y_train, y_pred_train)
    # report_test = classification_report(y_test, y_pred_test)
    # print('Training\n%s' % report_train)
    # print('Testing\n%s' % report_test)
    return y_propa, y_train



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', type=int, default=1, help='1 for local, 2 for sklearn model')
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    _model = args.model
    _random_state = args.random_state



    if _model == 1:
        logistic(_random_state)
    if _model == 2:
        sk_logistic(_random_state)