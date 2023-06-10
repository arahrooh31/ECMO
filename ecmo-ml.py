# Dependencies
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the data
df = pd.read_csv("ECMO_data_processed.csv")
df.head()

# Separate the features (X) and the target variable (y)
X = df.drop('outcome', axis=1)
y = df['outcome']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Split the existing test set into a new test set and a holdout set
# X_test, X_holdout, y_test, y_holdout = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# # Scale the features using StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Scale the features using MinMacScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a list to store the results
results = []

# XGBoost Model
XGB_model = xgb.XGBClassifier()
XGB_model.fit(X_train_scaled, y_train)
y_pred = XGB_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
y_prob = XGB_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('xgboost_roc.png')
#plt.show()
scores = cross_val_score(XGB_model, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())
results.append(['XGBoost', accuracy, precision, recall, f1, roc_auc, scores, scores.mean(), scores.std()])


# Handle missing values
imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)


# Handle missing values using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)


# SVM Model
SVM_model = SVC(probability=True)
SVM_model.fit(X_train_imputed, y_train)
y_pred = SVM_model.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
y_prob = SVM_model.predict_proba(X_test_imputed)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('svm_roc.png')
#plt.show()
scores = cross_val_score(SVM_model, X_train_imputed, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())
results.append(['SVM', accuracy, precision, recall, f1, roc_auc, scores, scores.mean(), scores.std()])

# Random Forest Model
RF_model = RandomForestClassifier()
RF_model.fit(X_train_imputed, y_train)
y_pred = RF_model.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

y_prob = RF_model.predict_proba(X_test_imputed)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest  Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('random_forest_roc.png')
#plt.show()

scores = cross_val_score(RF_model, X_train_imputed, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())
results.append(['Random Forest', accuracy, precision, recall, f1, roc_auc, scores, scores.mean(), scores.std()])

importance = RF_model.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(importance)

plt.figure()
plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance')
plt.tight_layout()  # Ensures that the labels fit within the figure
plt.savefig('random_forest_feature_importance.png')

importance = RF_model.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(importance)[-30:]  # Get the indices of the 20 most important features

indices = np.argsort(importance)[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature '{X_train.columns[indices[f]]}' ({importance[indices[f]]})")

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance (Top 30)')
plt.tight_layout()  # Ensures that the labels fit within the figure
plt.savefig('random_forest_feature_importance_top30.png')

# Logistic Regression Model
LR_model = LogisticRegression()
LR_model.fit(X_train_imputed, y_train)
y_pred = LR_model.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
y_prob = LR_model.predict_proba(X_test_imputed)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Receiver Operating Characteristic')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('logistic_regression_roc.png')
scores = cross_val_score(LR_model, X_train_imputed, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())
results.append(['Logistic Regression', accuracy, precision, recall, f1, roc_auc, scores, scores.mean(), scores.std()])



# Create a DataFrame from the results list
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Cross-validation scores', 'Mean accuracy', 'Standard deviation' ])
results_df



# Create a DataFrame to store the results
error_analysis_df = pd.DataFrame(index=y_test.index)

# Add the actual labels
error_analysis_df['Actual'] = y_test

# XGBoost Model
error_analysis_df['XGBoost_Predicted'] = XGB_model.predict(X_test_scaled)
error_analysis_df['XGBoost_Correct'] = error_analysis_df['Actual'] == error_analysis_df['XGBoost_Predicted']

# SVM Model
error_analysis_df['SVM_Predicted'] = SVM_model.predict(X_test_imputed)
error_analysis_df['SVM_Correct'] = error_analysis_df['Actual'] == error_analysis_df['SVM_Predicted']

# Random Forest Model
error_analysis_df['RandomForest_Predicted'] = RF_model.predict(X_test_imputed)
error_analysis_df['RandomForest_Correct'] = error_analysis_df['Actual'] == error_analysis_df['RandomForest_Predicted']

# Logistic Regression Model
error_analysis_df['LogisticRegression_Predicted'] = LR_model.predict(X_test_imputed)
error_analysis_df['LogisticRegression_Correct'] = error_analysis_df['Actual'] == error_analysis_df['LogisticRegression_Predicted']

# Display the error analysis DataFrame
print(error_analysis_df)
error_analysis_df.to_csv('error_analysis.csv', index=False)


# Count the number of correct and incorrect predictions for each model
xgb_correct_count = error_analysis_df['XGBoost_Correct'].sum()
xgb_incorrect_count = len(error_analysis_df) - xgb_correct_count

svm_correct_count = error_analysis_df['SVM_Correct'].sum()
svm_incorrect_count = len(error_analysis_df) - svm_correct_count

rf_correct_count = error_analysis_df['RandomForest_Correct'].sum()
rf_incorrect_count = len(error_analysis_df) - rf_correct_count

lr_correct_count = error_analysis_df['LogisticRegression_Correct'].sum()
lr_incorrect_count = len(error_analysis_df) - lr_correct_count

# Display the counts
print('XGBoost: Correct:', xgb_correct_count, 'Incorrect:', xgb_incorrect_count)
print('SVM: Correct:', svm_correct_count, 'Incorrect:', svm_incorrect_count)
print('Random Forest: Correct:', rf_correct_count, 'Incorrect:', rf_incorrect_count)
print('Logistic Regression: Correct:', lr_correct_count, 'Incorrect:', lr_incorrect_count)


# Create a DataFrame for the counts
counts_df = pd.DataFrame(columns=['Model', 'Correct', 'Incorrect'])

# Add the counts for each model
counts_df.loc[0] = ['XGBoost', xgb_correct_count, xgb_incorrect_count]
counts_df.loc[1] = ['SVM', svm_correct_count, svm_incorrect_count]
counts_df.loc[2] = ['Random Forest', rf_correct_count, rf_incorrect_count]
counts_df.loc[3] = ['Logistic Regression', lr_correct_count, lr_incorrect_count]

# Display the counts DataFrame
print(counts_df)



# Count the cases where all models got the predictions right
all_models_correct = error_analysis_df[
    (error_analysis_df['XGBoost_Correct'] == 1) &
    (error_analysis_df['SVM_Correct'] == 1) &
    (error_analysis_df['RandomForest_Correct'] == 1) &
    (error_analysis_df['LogisticRegression_Correct'] == 1)
]

# Count the cases where all models got the predictions wrong
all_models_wrong = error_analysis_df[
    (error_analysis_df['XGBoost_Correct'] == 0) &
    (error_analysis_df['SVM_Correct'] == 0) &
    (error_analysis_df['RandomForest_Correct'] == 0) &
    (error_analysis_df['LogisticRegression_Correct'] == 0)
]

# Get the counts
all_correct_count = len(all_models_correct)
all_wrong_count = len(all_models_wrong)

# Display the counts
print('Cases where all models got the predictions right:', all_correct_count)
print('Cases where all models got the predictions wrong:', all_wrong_count)


# Identify cases where all models got the prediction right
all_models_correct = error_analysis_df[['XGBoost_Correct', 'SVM_Correct', 'RandomForest_Correct', 'LogisticRegression_Correct']].all(axis=1)

# Identify cases where all models got the prediction wrong
all_models_wrong = error_analysis_df[['XGBoost_Correct', 'SVM_Correct', 'RandomForest_Correct', 'LogisticRegression_Correct']].all(axis=1).apply(lambda x: not x)

# Display the cases where all models got the prediction right
all_models_correct_cases = error_analysis_df[all_models_correct]
print('Cases where all models got the prediction right:')
print(all_models_correct_cases)

# Display the cases where all models got the prediction wrong
all_models_wrong_cases = error_analysis_df[all_models_wrong]
print('Cases where all models got the prediction wrong:')
print(all_models_wrong_cases)


