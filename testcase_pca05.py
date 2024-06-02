import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load data from Excel file
file_path = "selected_data.xlsx"
df = pd.read_excel(file_path)

# Separate features from the target variable if applicable
# Replace 'target_column_name' with the actual name of your target column
X = df.drop(columns=['Completed'])  # Remove the target column if applicable
y = df['Completed']  # Assuming 'Completed' is the target column

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(X_scaled)

# Determine the number of principal components to retain
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1  # Retain components that explain at least 95% of the variance

# Re-apply PCA with the determined number of components
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_scaled)

# Display the selected features
selected_features = X.columns.to_numpy()[pca.components_[:n_components].argsort()[::-1][:n_components]]
print("Selected Features after PCA:")
print(selected_features)

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid()
plt.show()

# Biplot
def biplot(score, coeff, labels=None):
    plt.figure(figsize=(10, 6))
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c=df['Completed'] if 'Completed' in df.columns else None)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "PC" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()

biplot(principal_components[:, :2], np.transpose(pca.components_[:2, :]), labels=X.columns)

# Create a DataFrame for the principal components
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])
principal_df['Completed'] = y

# Add selected features to the DataFrame
for i, feature in enumerate(selected_features):
    principal_df[feature] = X[feature]

# Save the transformed features to Excel
principal_df.to_excel("transformed_features.xlsx", index=False)

# Heatmap of correlation between original features and principal components
correlation_matrix = np.corrcoef(X_scaled.T, principal_components.T)[:X_scaled.shape[1], X_scaled.shape[1]:]
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=X.columns, yticklabels=[f'PC{i}' for i in range(1, n_components + 1)])
plt.title('Correlation Heatmap')
plt.show()

# Cross-validation
gbdt = GradientBoostingClassifier()

# Cross-validation scores for Gradient Boosting Decision Trees
gbdt_scores = cross_val_score(gbdt, principal_components, y, cv=5, scoring='accuracy')
print("Gradient Boosting Decision Trees CV Scores:", gbdt_scores)
print("Mean Accuracy (Gradient Boosting Decision Trees):", gbdt_scores.mean())

# Model Performance Comparison
# Train model on original features
gbdt_original = GradientBoostingClassifier()
gbdt_original.fit(X_scaled, y)

# Predictions using original features
gbdt_pred_original = gbdt_original.predict(X_scaled)

# Accuracy score using original features
accuracy_gbdt_original = accuracy_score(y, gbdt_pred_original)
print("Accuracy (Gradient Boosting Decision Trees - Original):", accuracy_gbdt_original)

# Train model on transformed features
gbdt_pca = GradientBoostingClassifier()
gbdt_pca.fit(principal_components, y)

# Predictions using transformed features
gbdt_pred_pca = gbdt_pca.predict(principal_components)

# Accuracy score using


# Accuracy score using transformed features
accuracy_gbdt_pca = accuracy_score(y, gbdt_pred_pca)
print("Accuracy (Gradient Boosting Decision Trees - PCA):", accuracy_gbdt_pca)
