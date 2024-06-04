import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Read data from Excel file
excel_file = "encoded_data.xlsx"
data = pd.read_excel(excel_file)

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can use other strategies too
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Separate features and target variable
features = data_imputed.iloc[:, :-1]  # Assuming the last column is the target variable
target = data_imputed.iloc[:, -1]

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the features data
normalized_features = scaler.fit_transform(features)

# Convert normalized features back to a DataFrame
normalized_data = pd.DataFrame(normalized_features, columns=features.columns)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(normalized_data, target)

# Save the balanced data to an Excel file
balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=features.columns), pd.Series(y_resampled, name=target.name)], axis=1)
output_excel_file = "balanced_data.xlsx"
balanced_data.to_excel(output_excel_file, index=False)

# Visualization before normalization
plt.figure(figsize=(12, 6))
for i, column in enumerate(features.columns):
    plt.subplot(2, len(features.columns), i+1)
    sns.histplot(data[column], kde=True)
    plt.title("Before Normalization")
    plt.xlabel("")
    plt.ylabel("")

# Visualization after normalization and SMOTE
for i, column in enumerate(normalized_data.columns):
    plt.subplot(2, len(features.columns), len(features.columns)+i+1)
    sns.histplot(data=X_resampled, x=column, kde=True)
    plt.title("After Normalization and SMOTE")
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.show()
