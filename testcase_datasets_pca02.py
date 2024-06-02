import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Excel
file_path = 'selected_data.xlsx'  # Replace with your file path
sheet_name = 'Sheet1'  # Replace with your sheet name if necessary
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Specify the target feature column name
target_feature = 'Completed'  # Replace with the name of your target feature column

# Extract all feature columns
selected_features = data.columns[data.columns != target_feature]

# Separate selected features and target
features = data[selected_features]
target = data[target_feature]

# Standardize the selected features (mean=0, variance=1)
features_standardized = (features - features.mean()) / features.std()

# Compute the covariance matrix
cov_matrix = np.cov(features_standardized.T)

# Perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Display the eigenvalues
print("Eigenvalues:")
print(sorted_eigenvalues)

# Display the eigenvectors
print("\nEigenvectors:")
print(sorted_eigenvectors)

# Explained variance
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

# Cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance)

# Plot the scree plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', label='Explained Variance')
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, 's-', label='Cumulative Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')
plt.legend()
for i, v in enumerate(selected_features):
    plt.text(i + 1, explained_variance[i], v, ha='right', va='bottom')
plt.grid(True)
plt.show()

# Project the data onto the principal components
projected_data = np.dot(features_standardized, sorted_eigenvectors)

# Create a DataFrame for the projected data
projected_df = pd.DataFrame(projected_data, columns=[f'PC{i+1}' for i in range(projected_data.shape[1])])
projected_df[target_feature] = target

# Save the projected data to a new Excel file
output_file_path = 'projected_data_with_target.xlsx'
projected_df.to_excel(output_file_path, index=False)

# Include the target feature in the selected features dataset
selected_features = list(selected_features) + [target_feature]
selected_features_df = data[selected_features]

# Save the selected features dataset to a new Excel file
selected_features_file_path = 'selected_features.xlsx'
selected_features_df.to_excel(selected_features_file_path, index=False)
