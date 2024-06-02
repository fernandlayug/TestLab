import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Excel file
file_path = "selected_data.xlsx"
df = pd.read_excel(file_path)

# Separate features from the target variable if applicable
# Replace 'target_column_name' with the actual name of your target column
X = df.drop(columns=['Completed'])  # Remove the target column if applicable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(X_scaled)

# Scree plot to determine the number of principal components to retain
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid()
plt.show()

# Determine the number of principal components to retain
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1  # Retain components that explain at least 95% of the variance

# Re-apply PCA with the determined number of components
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame for the principal components
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])

# Concatenate the principal components DataFrame with the target variable if applicable
# Replace 'target_column_name' with the actual name of your target column
# Replace 'target_variable' with the actual target variable from your DataFrame
if 'Completed' in df.columns:
    principal_df['0'] = df['Completed']

# Save the transformed features to Excel
principal_df.to_excel("transformed_features.xlsx", index=False)

# Heatmap of correlation between original features and principal components
correlation_matrix = np.corrcoef(X_scaled.T, principal_components.T)[:X_scaled.shape[1], X_scaled.shape[1]:]
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=X.columns, yticklabels=[f'PC{i}' for i in range(1, n_components + 1)])
plt.title('Correlation Heatmap')
plt.show()

# Get the names of the selected features
selected_features = X.columns[np.abs(correlation_matrix).argmax(axis=0)]
print("Selected Features:")
print(selected_features)
