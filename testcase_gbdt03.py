import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Load your dataset from Excel (replace 'your_dataset.xlsx' with your Excel file)
data = pd.read_excel('transformed_features.xlsx')

# Separate features and target variable
X = data.drop(columns=['Completed'])  # Features
y = data['Completed']  # Target variable

# Split the data into training and testing sets (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preparation: Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Gradient Boosting Classifier
gbdt = GradientBoostingClassifier()

# Define the hyperparameters grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=gbdt, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

# Use the best parameters to train the model
best_gbdt = GradientBoostingClassifier(**best_params)
best_gbdt.fit(X_train_scaled, y_train)

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_gbdt, X_train_scaled, y_train, cv=kf)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Evaluation on the hold-out test set
test_accuracy = best_gbdt.score(X_test_scaled, y_test)
print("Testing Accuracy:", test_accuracy)

# Visualize cross-validation scores
plt.figure(figsize=(8, 6))
sns.barplot(x=[f"Fold {i+1}" for i in range(len(cv_scores))], y=cv_scores)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores')
plt.ylim(0.8, 1.0)
plt.show()

# Visualize confusion matrix for testing set
y_pred = best_gbdt.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Testing Set')
plt.show()
