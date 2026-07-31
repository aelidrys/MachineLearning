from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from credit_fraud_utils_data import load_data, preprocess_eval_data
import argparse
import pickle



# Load the trained model from a file
def load_model_data(model_path):
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        raise FileNotFoundError(f"Error loading model: {e}")


# Evaluate the model on the given dataset
def evaluate_model(model, threshold, X, y, model_name, dataset_name="train"):
    print("\n-------------------------------------------------------------------------------")
    
    print(f"Evaluating the {model_name} model by {dataset_name} dataset")
    print(f"Threshold: {threshold:.2f}")
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    if y_proba is not None:
        auc_score = roc_auc_score(y, y_proba)
        print(f"\nROC AUC Score: {auc_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Evaluation Script")
    parser.add_argument('--data', type=str, default='data/test.csv', help='Path to the test data')
    parser.add_argument('--model_path', type=str, default='model.pkl', help='Path to the trained model')
    args = parser.parse_args()

    # Load the test data
    data = load_data(args.data)

    # Load the trained model
    model_data = load_model_data(args.model_path)

    # Preprocess the test data
    bounds = model_data.get('bounds', [])
    X_test, Y_test = preprocess_eval_data(data, bounds)

    # Evaluate the model on the test data_set
    dataset_name = args.data.split('/')[-1].split('.')[0]
    threshold = model_data.get('threshold', 0.5)
    model_name = model_data.get('model_name', 'Unknown')
    model = model_data.get('model', None)
    if model is None:
        raise ValueError("Model not found in the loaded model data.")
    evaluate_model(model, threshold, X_test, Y_test, model_name, dataset_name)


if __name__ == "__main__":
    main()