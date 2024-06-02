import pandas as pd
from sklearn.model_selection import train_test_split

# Load the Excel file into a DataFrame
excel_file = "transformed_features.xlsx"  # Replace "your_dataset.xlsx" with the path to your Excel file
data = pd.read_excel(excel_file)

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['0'])  # Replace 'target_column' with the name of your target column
y = data['0']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust test_size as needed

# Save the training and testing sets to new Excel files
X_train.to_excel("X_train.xlsx", index=False)
X_test.to_excel("X_test.xlsx", index=False)
y_train.to_excel("y_train.xlsx", index=False)
y_test.to_excel("y_test.xlsx", index=False)
