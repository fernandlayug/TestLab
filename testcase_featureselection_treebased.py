import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load data from Excel file
df = pd.read_excel('balanced_data.xlsx')

# Split data into features (X) and target variable (y)
X = df.drop(columns=['Completed'])
y = df['Completed']

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance from RandomForest')
plt.gca().invert_yaxis()  # Invert y-axis to display features from top to bottom
plt.show()
