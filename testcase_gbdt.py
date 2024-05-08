import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV
data = pd.read_excel('ALSResponses05.xlsx')

# Separate features and target variable
X = data.drop('Completed', axis=1)  # Adjust 'target_column' to your target variable name
y = data['Completed']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create GBDT classifier
gbdt_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit classifier to training data
gbdt_classifier.fit(X_train, y_train)

# Make predictions
predictions = gbdt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
