from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score



def evaluate_model(model, threshold, X, y, dataset_name="train"):
    
    print("\n-------------------------------------------------------------------------------")
    print(f"Evaluating the model by {dataset_name} dataset")
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    if y_proba is not None:
        auc_score = roc_auc_score(y, y_proba)
        print(f"\nROC AUC Score: {auc_score:.4f}")