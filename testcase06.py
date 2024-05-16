# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'dataset.csv' with your dataset file)
dataset = pd.read_excel('ALSResponses05.xlsx')

# Preprocessing the dataset
# Assuming your dataset has features and a target variable named 'features' and 'target' respectively
X = dataset.drop('Completed', axis=1)
y = dataset['Completed']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Gradient Boosting Classifier
gbdt_model = GradientBoostingClassifier()

# Defining hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Setting up GridSearchCV
grid_search = GridSearchCV(estimator=gbdt_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)

# Training the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best Parameters:", grid_search.best_params_)

# Making predictions using the best model
best_gbdt_model = grid_search.best_estimator_
y_pred = best_gbdt_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Printing classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Getting the predicted probabilities for the positive class
y_probs = best_gbdt_model.predict_proba(X_test)[:, 1]

# Calculating ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculating AUC score
auc_score = roc_auc_score(y_test, y_probs)
print("AUC Score:", auc_score)

# Plotting ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plotting Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Retained', 'Dropped Out'], yticklabels=['Retained', 'Dropped Out'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
