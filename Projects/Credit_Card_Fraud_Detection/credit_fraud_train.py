import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from credit_fraud_utils_data import load_data, split_data, preprocess_data, preprocess_eval_data, save_model, sampling_data
from credit_fraud_utils_eval import evaluate_model
from sklearn.metrics import precision_recall_curve
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from credit_fraud_visualization import pr_curve, precisionVsRecall



def train_model(X, Y, model_name):
    print("Training START")
    counter = Counter(Y)
    print(f"class0 size: {counter[0]}, class1 size: {counter[1]}")
    if model_name == 'logisticReg':
        model = LogisticRegression()
    elif model_name == 'randomForest':
        model = RandomForestClassifier()
    elif model_name == 'neuralNet':
        model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=5000)
    elif model_name == 'voting':
        model1 = LogisticRegression(class_weight='balanced')
        model2 = RandomForestClassifier()
        model3 = MLPClassifier(hidden_layer_sizes=(10,), max_iter=5000)
        model = VotingClassifier(estimators=[('logr', model1), ('rf', model2), ('nn', model3)], voting='soft')
    elif model_name == 'xgboost':
        model = XGBClassifier(n_estimators=5, max_depth=25, learning_rate=0.01, objective='binary:logistic')
    elif model_name == 'catboost':
        model = CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=6, verbose=0)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(n_estimators=5, learning_rate=0.01, max_depth=6, objective='binary')
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    model.fit(X, Y)
    print("Training FINISH")
    return model


def best_threshold(model, X, Y):
    
    y_propa =  model.predict_proba(X)[:, 1]
    precesions, recalls, thresholds = precision_recall_curve(Y, y_propa)
    

    # Visualize the precision-recall curve
    pr_curve(precesions, recalls, thresholds)
    precisionVsRecall(precesions, recalls)
    f1_scoures = 2 * (precesions * recalls) / (precesions + recalls)
    best_index = np.argmax(f1_scoures)
    best_threshold = thresholds[best_index]
    f1_scoure = f1_scoures[best_index]
    print(f"best_threshold: {best_threshold:.2f}, f1_scoure: {f1_scoure:.2f}")
    return best_threshold


def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training Script")
    parser.add_argument('--data', type=str, default='data/train.csv', help='Path to the training data')
    parser.add_argument('--model', type=str, default='logisticReg', 
            help='logisticReg to train model using Logistic Regression'
                 'randomForest to train model using Random Forest'
                 'neuralNet to train model using Neural Network'
                 'voting to train model using Voting Classifier'
                 'xgboost to train model using XGBoost'
                 'catboost to train model using CatBoost'
                 'lightgbm to train model using LightGBM'
                 )
    parser.add_argument('--save_path', type=str, default='model.pkl', help='Path to save the trained model')
    parser.add_argument('--save_model', type=int, default=0, help='1 to save the trained model or 0 to not save it')
    parser.add_argument('--eval', type=int, default=0, help='1 to evaluate the model on the test dataset or 0 to not evaluate it')
    args = parser.parse_args()

    # Load the training data
    data = load_data(args.data)

    # Preprocess the data
    X_train, X_val, Y_train, Y_val, bounds = preprocess_data(data)

    # Sample the training data to handle class imbalance
    # X_train, Y_train = sampling_data(X_train, Y_train)

    # Train the model
    model = train_model(X_train, Y_train, args.model)
    
    # Tuning the threshold for the model
    threshold = best_threshold(model, X_train, Y_train)
    
    # Evaluate the model on the training dataset
    evaluate_model(model, threshold, X_train, Y_train, args.model, "train")

    # Evaluate the model on the validation dataset
    evaluate_model(model, threshold, X_val, Y_val, args.model, "validation")
    
    # Evaluate the model on the test dataset
    if args.eval == 1:
        data = load_data('data/test.csv')
        X, Y = preprocess_eval_data(data)
        evaluate_model(model, threshold, X, Y, args.model, "test")
    

    # Save the trained model
    if args.save_model == 1:
        save_model(model, threshold, args.model, bounds, args.save_path)


if __name__ == "__main__":
    main()