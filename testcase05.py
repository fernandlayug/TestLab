# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
# Assuming you have a CSV file with features and labels
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_excel('ALSResponses05.xlsx')

# Split the data into features and target variable
X = data.drop('Completed', axis=1) # Assuming 'dropout' is the label column
y = data['Completed']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=3, scoring='accuracy')

# Perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:")
print(grid_search.best_params_)
print()

# Get the best model
best_gb_classifier = grid_search.best_estimator_

# Make predictions
y_pred = best_gb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report with precision, recall, and F1-score
print(classification_report(y_test, y_pred))

# You can also use the trained model to predict on new data
# new_data = ... # load or create new data
# new_predictions = best_gb_classifier.predict(new_data)
