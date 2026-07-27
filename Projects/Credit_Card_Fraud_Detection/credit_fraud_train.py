import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from credit_fraud_utils_data import load_data, split_data, preprocess_data, preprocess_test_data
from credit_fraud_utils_data import save_model, sampling_data
from credit_fraud_utils_eval import evaluate_model
from sklearn.metrics import precision_recall_curve
from collections import Counter
from sklearn.neural_network import MLPClassifier



def train_model(X, Y, model_name):
    print("Training START")
    counter = Counter(Y)
    print(f"class0 size: {counter[0]}, class1 size: {counter[1]}")
    if model_name == 'logisticReg':
        model = LogisticRegression()
    elif model_name == 'randomForest':
        model = RandomForestClassifier()
    elif model_name == 'neuralNet':
        model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=5000)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    model.fit(X, Y)
    print("Training FINISH")
    return model


def best_threshold(model, X, Y):
    
    y_propa =  model.predict_proba(X)[:, 1]
    precesions, recalls, thresholds = precision_recall_curve(Y, y_propa)
    
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
                 'voting to train model using Voting Classifier')
    parser.add_argument('--save_path', type=str, default='model.pkl', help='Path to save the trained model')
    parser.add_argument('--save_model', type=int, default=0, help='1 to save the trained model or 0 to not save it')
    args = parser.parse_args()

    # Load the training data
    data = load_data(args.data)

    # Preprocess the data
    X_train, X_val, Y_train, Y_val = preprocess_data(data)

    # Train the model
    model = train_model(X_train, Y_train, args.model)
    
    # Tuning the threshold for the model
    threshold = best_threshold(model, X_train, Y_train)
    
    # Evaluate the model on the training dataset
    evaluate_model(model, threshold, X_train, Y_train, "train")

    # Evaluate the model on the validation dataset
    evaluate_model(model, threshold, X_val, Y_val, "validation")
    
    # Evaluate the model on the test dataset
    # data = load_data('data/test.csv')
    # X, Y = preprocess_test_data(data)
    # evaluate_model(model, threshold, X, Y, "test")
    

    # Save the trained model
    if args.save_model == 1:
        save_model(model, args.save_path)
        print(f"Model saved to {args.save_path}")
    
    
if __name__ == "__main__":
    main()