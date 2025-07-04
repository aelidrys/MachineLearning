import matplotlib.pyplot as plt
import numpy as np
from model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model', type=int, default=1, help='1 for local, 2 for sklearn model')
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()
_model = args.model
_random_state = args.random_state


data = load_breast_cancer()

def data_preparation(add_intercept=True, random_state=42):
    X, t = data.data, data.target.reshape((-1, 1))
    X = MinMaxScaler().fit_transform(X)
    
    if add_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.3, shuffle=True,
        stratify=t ,random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def logistic():
    X_train, X_test, y_train, y_test = data_preparation(random_state=_random_state)

    model = LogisticRegression()
    model.fit(X_train, y_train, _step_size=1, _precision=1e-7, _max_iter=100000)

    # training
    print('Training results:')
    y_pred = model.predict(X_train)
    p_classes = (y_pred >= 0.5).astype(int)
    print(f'\t{np.count_nonzero(p_classes == y_train)}/{y_train.size}')
    print(f'\taccuracy: {accuracy_score(y_train, p_classes)}')
    # print(f'Train classification_report: {classification_report(y_train, p_classes)}')

    # testing
    print('----------------------------------------')
    print('Testing results:')
    y_test_pred = model.predict(X_test)
    p_test_classes = (y_test_pred >= 0.5).astype(int)
    print(f'\t{np.count_nonzero(p_test_classes == y_test)}/{y_test.size}')
    print(f'\taccuracy: {accuracy_score(y_test, p_test_classes)}')
    # print(f'Test classification_report: {classification_report(y_test, p_test_classes)}')


def sk_logistic():
    X_train, X_test, y_train, y_test = data_preparation(random_state=_random_state)
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_test_prop = model.predict_proba(X_test)[:, 1]

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print('Training accuracy: %.4f' % accuracy_train)
    print('Test accuracy:     %.4f' % accuracy_test)

    report_train = classification_report(y_train, y_pred_train)
    report_test = classification_report(y_test, y_pred_test)
    # print('Training\n%s' % report_train)
    # print('Testing\n%s' % report_test)



if __name__ == '__main__':
    if _model == 1:
        logistic()
    if _model == 2:
        sk_logistic()