import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Excel file
file_path = 'selected_data_1.xlsx'  # Update with your file path
data = pd.read_excel(file_path)

# Separate features from the target variable, if any
target_column = 'Completed'  # Update with the name of your target column, if any
X = data.drop(columns=[target_column])
y = data[target_column]

# Get the names of the features
feature_names = X.columns.tolist()

# Feature selection using SelectKBest
best_features_selector = SelectKBest(score_func=f_regression, k=5)  # Select the best 5 features
X_best = best_features_selector.fit_transform(X, y)
best_feature_indices = best_features_selector.get_support(indices=True)
best_feature_names = [feature_names[i] for i in best_feature_indices]

# Standardize the selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_best)

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

# Heatmap of the PCA loadings
plt.figure(figsize=(10, 6))
loadings = pca.components_.T
sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=[f'PC{i+1}' for i in range(loadings.shape[1])],
            yticklabels=best_feature_names, cbar_kws={'label': 'Loading Value'})
plt.title('PCA Loadings Heatmap')
plt.xlabel('Principal Components')
plt.ylabel('Features')
plt.show()
