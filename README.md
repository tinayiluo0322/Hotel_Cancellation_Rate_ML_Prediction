## Hotel Cancellation Prediction Model

This project aims to develop and fine-tune various machine learning models to predict hotel cancellations, as part of a Kaggle competition. By leveraging different algorithms and hyperparameter tuning techniques, we seek to identify the most effective model for this task, providing accurate predictions and valuable insights into hotel booking behaviors.

## Process on Model Selection and Hyperparameter Tuning

### Logistic Regression
The logistic regression algorithm, known for its simplicity and effectiveness in binary classification tasks, was chosen for predicting hotel cancellations. For hyperparameter tuning, I focused on the regularization strength (C) and opted for L1 regularization (lasso) to avoid overfitting and aid in feature selection. I employed a grid search over 20 logarithmically spaced values of C, ranging from \(10^{-4}\) to \(10^{4}\), using the 'saga' solver. The best C value was identified based on the AUC metric on the validation dataset. The final model was trained with this optimal C value and evaluated using AUC, ROC curve, Precision-Recall (PR) curve, and Average Precision (AP) score.

### K-Nearest Neighbors (KNN)
I applied the KNN model to predict hotel cancellations, focusing on finding the optimal number of neighbors (K) to balance the bias-variance trade-off. A grid search over K values from 1 to 50 was conducted, with model performance evaluated using the AUC score on the validation dataset. The optimal K value was selected, and the final KNN model was retrained and evaluated using AUC, ROC curve, PR curve, and AP score.

### Random Forest
The Random Forest model, suitable for handling tabular data, was tuned using RandomizedSearchCV. The hyperparameters explored included the number of trees (`n_estimators`), maximum depth (`max_depth`), minimum samples to split (`min_samples_split`), and minimum samples at leaf nodes (`min_samples_leaf`). After identifying the best parameters based on the AUC score across 5-fold cross-validation, the final Random Forest model was trained and evaluated using AUC, ROC curve, PR curve, and AP score.

### CatBoost
The CatBoost model, designed to handle categorical variables efficiently, was tuned using RandomizedSearchCV. The hyperparameters explored included the learning rate (`learning_rate`), depth of the trees (`depth`), and the number of trees (`iterations`). The best combination of hyperparameters was identified based on the AUC score. The final CatBoost model was trained and evaluated using AUC, ROC curve, PR curve, and AP score.

### XGBoost
XGBoost, a gradient-boosting decision tree algorithm, was tuned using RandomizedSearchCV. The hyperparameters explored included the number of trees (`n_estimators`), learning rate, maximum depth (`max_depth`), subsample, and colsample_bytree. The best parameters were selected based on the AUC score across 5-fold cross-validation. The final XGBoost model was trained and evaluated using AUC, ROC curve, PR curve, and AP score.

## Model Performance Evaluation
In the analysis of ROC curves, Random Forest, CatBoost, and XGBoost achieved exceptional performance, each with an AUC score around 0.96. XGBoost slightly edged out its competitors with an AUC score of approximately 0.963. Logistic Regression and KNN lagged, with AUC scores of 0.916 and 0.895, respectively. PR curves reinforced these findings, with Random Forest, CatBoost, and XGBoost each achieving an Average Precision (AP) score around 0.94. XGBoost led with an AP score of about 0.946. KNN and Logistic Regression had lower AP scores, with KNN at 0.842 and Logistic Regression at 0.873.

In terms of computational efficiency, KNN required more time for prediction (17.16 seconds) than for training (8.1 seconds), while Logistic Regression had the longest training time (999.59 seconds) but the quickest prediction time (0.34 seconds). CatBoost and XGBoost demonstrated exceptional efficiency, balancing quick training times (22.36 seconds for CatBoost and 34.98 seconds for XGBoost) with swift prediction capabilities (under 1 second for both).

## Conclusion
In conclusion, XGBoost emerged as the superior model, with the highest AUC and AP scores and good computational efficiency, making it highly suitable for predicting hotel cancellations. The comprehensive evaluation and tuning process across multiple algorithms ensured that the most effective and efficient model was selected, providing robust predictions and valuable insights into hotel booking behaviors. This project highlights the importance of model selection, hyperparameter tuning, and thorough evaluation in achieving high-performing machine learning models for real-world applications.
