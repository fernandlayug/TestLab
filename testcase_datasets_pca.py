import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset from Excel
data = pd.read_excel("selected_data.xlsx")

# Separate features (X) and target (y) if applicable
X = data.drop(columns=["Completed"])  # Adjust "target_column" to your target column name, if applicable
y = data["Completed"]  # Adjust "target_column" to your target column name, if applicable

# Standardize the features (optional but recommended for PCA)
X_standardized = (X - X.mean()) / X.std()

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Calculate the cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--')  # Optional: Add a threshold line for 95% variance
plt.show()

# Determine the number of components that explain at least 95% of the variance
n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1
print(f'Number of principal components that explain at least 95% of the variance: {n_components}')

# Perform PCA with the optimal number of components
pca_optimal = PCA(n_components=n_components)
X_pca_optimal = pca_optimal.fit_transform(X_standardized)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=X_pca_optimal, columns=[f"PC{i+1}" for i in range(n_components)])

# If you have target classes, add them to the PCA DataFrame
# pca_df["target"] = y

# Plot PCA results for the first two components
plt.figure(figsize=(10, 6))
plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=y, cmap='viridis')  # Use 'viridis' colormap or adjust as needed
plt.title('PCA of Dataset (First Two Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Target')
plt.grid(True)
plt.show()

# Display the features that contribute most to each principal component
loading_scores = pd.DataFrame(pca_optimal.components_.T, columns=[f"PC{i+1}" for i in range(n_components)], index=X.columns)
print("Loading scores for each principal component:")
print(loading_scores)

# Identify the features with the highest absolute loading scores for the first two principal components
top_features_pc1 = loading_scores['PC1'].abs().sort_values(ascending=False).head(10)
top_features_pc2 = loading_scores['PC2'].abs().sort_values(ascending=False).head(10)

print("\nTop features contributing to Principal Component 1:")
print(top_features_pc1)

print("\nTop features contributing to Principal Component 2:")
print(top_features_pc2)
