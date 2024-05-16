import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data from Excel file
file_path = 'ALSResponses05.xlsx'  # Update with your file path
data = pd.read_excel(file_path)

# Separate features from the target variable, if any
X = data.drop(columns=['Completed'])  # Update with the name of your target column, if any

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(8, 6))
components = range(1, len(pca.explained_variance_ratio_) + 1)
plt.bar(components, pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
# Add labels
for i, ratio in enumerate(pca.explained_variance_ratio_):
    plt.text(components[i], ratio + 0.01, f'{ratio:.2f}', ha='center')
plt.show()

# You can choose a number of components based on the explained variance ratio plot or specify it directly
# For example, to choose the number of components that explain 95% of the variance:
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components to explain 95% variance: {n_components}")

# Re-fit PCA with chosen number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Output the principal components
print("Principal Components:")
print(X_pca)


# Interpretation
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Mean
X_mean = X.mean()
 
# Standard deviation
X_std = X.std()
 
# Standardization
Z = (X - X_mean) / X_std

# covariance
c = Z.cov()


# Plot the covariance matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(c)
plt.show()

# Optionally, you can save the transformed data to a new Excel file
pca_data = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, n_components+1)])
pca_data.to_excel('pca_data.xlsx', index=False)
