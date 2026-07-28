# Credit Card Fraud Detection
---
### [Project description in Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) 

## OverView
- ### Exploration Data Analyses
    - #### [my EDA](./exploration_data.ipynb)

- ### Data Preprocessing
    - #### Select importante featurs depend on random forest model and pairplot and correlation matrix.

    - #### Treat outliers using Intere Quartile Range.

    - #### Split the data to train and val sets.

- ### Train the model
    - #### Choose the model depend on the user input or use logistic regession as a default model.

    - #### Choose the hyper parameters by experemnt.

    - #### Fit the selected model
    
    - #### Use precision and recall function to choose the best threshold that have the highest F1_scoure.

    - #### Save the model and threshold and the preprocessing data parameters to apply it to in the test phase


- ### Evaluate the model