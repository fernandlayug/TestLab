import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Read data from Excel file
excel_file = "encoded_data.xlsx"
data = pd.read_excel(excel_file)

# Separate features and target variable
features = data.iloc[:, :-1]  # Assuming the last column is the target variable
target = data.iloc[:, -1]

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the features data
normalized_features = scaler.fit_transform(features)

# Convert normalized features back to a DataFrame
normalized_data = pd.DataFrame(normalized_features, columns=features.columns)

# Concatenate normalized features with target variable
normalized_data_with_target = pd.concat([normalized_data, target], axis=1)

# Save the normalized data to an Excel file
output_excel_file = "normalized_data.xlsx"
normalized_data_with_target.to_excel(output_excel_file, index=False)

# Visualization before normalization
plt.figure(figsize=(12, 6))
for i, column in enumerate(features.columns):
    plt.subplot(2, len(features.columns), i+1)
    sns.histplot(data[column], kde=True)
    plt.title("Before Normalization")
    plt.xlabel("")
    plt.ylabel("")

# Visualization after normalization
for i, column in enumerate(normalized_data.columns):
    plt.subplot(2, len(features.columns), len(features.columns)+i+1)
    sns.histplot(data=normalized_data, x=column, kde=True)
    plt.title("After Normalization")
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.show()
