import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your dataset from a CSV file
df = pd.read_csv('ALSResponses.csv')

# Separate features (X)
X = df.drop(columns=['Completed'])  # Features

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Create a dataframe from the PCA results
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA')
plt.show()

# Variance explained by each principal component
print('Explained variance ratio:', pca.explained_variance_ratio_)
