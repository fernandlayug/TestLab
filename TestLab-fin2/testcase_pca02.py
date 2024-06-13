import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Read data from Excel file into pandas DataFrame
df = pd.read_excel("selected_data_1.xlsx")

# Step 2: Exclude the target column
target_column = "Completed"
df_features = df.drop(columns=[target_column])

# Step 3: Preprocess the data if necessary
# For example, handle missing values or encode categorical variables

# Step 4: Standardize the features
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df_features)

# Step 5: Apply PCA
pca = PCA()
pca.fit(df_standardized)


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


# Step 6: Analyze explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Scree plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid',
         label='Cumulative explained variance')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Determine the number of principal components to retain
n_components = len(cumulative_variance_ratio[cumulative_variance_ratio <= 0.95])

# Step 7: Transform original features into new feature space using selected principal components
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(df_standardized)

# Step 8: Analyze loadings of each principal component
loadings = pca.components_

# Get the absolute values of loadings for each feature
abs_loadings = np.abs(loadings)

# Determine the most important feature for each principal component
most_important_features = abs_loadings.argmax(axis=1)

# Get the names of the selected features
selected_features = df_features.columns[most_important_features]

print("Selected features:")
print(selected_features)

# Create a DataFrame for loadings
loadings_df = pd.DataFrame(loadings, columns=df_features.columns)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('PCA Loadings Heatmap')
plt.xlabel('Original Features')
plt.ylabel('Principal Components')
plt.show()
