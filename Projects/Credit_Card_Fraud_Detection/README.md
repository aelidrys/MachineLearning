# Credit Card Fraud Detection *(LogisticReg, RandomForest, XGBoost,...)*
---
### [Project description in Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) 

## OverView
- ### Definition:
    #### It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
- ### Dataset:
    #### The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.17% of all transactions.
- ### Exploration Data Analyses
    - #### [my EDA](./exploration_data.ipynb)
---

## Workflow
- ### Data Preprocessing steps
    - #### Select importante featurs depend on random forest model and pairplot and correlation matrix.
        ![](./statics/output.png)
        #### Selected features ['V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']

    - #### Treat outliers using Intere Quartile Range.

    - #### Split the data to train and val sets.

- ### Train the model
    - #### Choose the model depend on the user input or use logistic regession as a default model.

    - #### Choose the hyper parameters by experemnt.

    - #### Fit the selected model
    
    - #### Use precision and recall function to choose the best threshold that have the highest F1_scoure.

    - #### Save the model and threshold and the preprocessing data parameters to apply it to in the test phase


- ### Evaluate the model
    - ### Preprocessing test dataset
        - #### Aplly the same preprocessing process to the test dataset with the extracted parameters from the train dataset without extracte new parameters from the test dataset.

    - ### Report F1_scoure and classification_report, auc 