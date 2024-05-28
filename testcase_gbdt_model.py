import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
file_path = 'ALSResponses05.xlsx'
sheet_name = 'ALSResponses'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Assume the last column is the target variable and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode the target variable if it is categorical
if y.dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model to a .pkl file
joblib.dump(model, 'gradient_boosting_model2.pkl')

# Later, load the model from the .pkl file
loaded_model = joblib.load('gradient_boosting_model2.pkl')

# Predict on the test set and evaluate the loaded model
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
