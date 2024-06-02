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

# Plot eigenvectors
plt.figure(figsize=(10, 5))
plt.bar(range(len(sorted_eigenvalues)), sorted_eigenvalues)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues')
plt.grid(True)
plt.show()

# Plot covariance matrix heatmap with eigenvectors
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Covariance Matrix with Eigenvectors')
for i, v in enumerate(sorted_eigenvectors.T):
    plt.arrow(0, 0, v[0], v[1], color='black', width=0.05, head_width=0.2)
    plt.text(v[0], v[1], f'PC {i+1}', ha='right', va='bottom', fontsize=8)
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()
