import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data from Excel into a pandas DataFrame
df = pd.read_excel('ALSResponses05.xlsx')  # Replace 'your_dataset.xlsx' with the path to your Excel file


# Assuming your data is numerical and you want to perform PCA on all columns except the target column (if any)
X = df.drop(columns=['Completed'])  # Adjust 'target_column_name' to the name of your target column, if any


# Apply PCA
pca = PCA(n_components=8)  # You can choose the number of components you want
X_pca = pca.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA')
plt.show()

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


# eigenvalues, eigenvectors = np.linalg.eig(c)
# print('Eigen values:\n', eigenvalues)
# print('Eigen values Shape:', eigenvalues.shape)
# print('Eigen Vector Shape:', eigenvectors.shape)


# # Index the eigenvalues in descending order 
# idx = eigenvalues.argsort()[::-1]

# # Sort the eigenvalues in descending order 
# eigenvalues = eigenvalues[idx]

# # sort the corresponding eigenvectors accordingly
# eigenvectors = eigenvectors[:,idx]


# explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
# explained_var

# n_components = np.argmax(explained_var >= 0.40) + 1
# n_components


# # PCA component or unit matrix
# u = eigenvectors[:,:n_components]
# pca_component = pd.DataFrame(u,
# 							index = cancer['feature_names'],
# 							columns = ['PC1','PC2']
# 							)

# # plotting heatmap
# plt.figure(figsize =(5, 7))
# sns.heatmap(pca_component)
# plt.title('PCA Component')
# plt.show()

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(c)

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Calculate explained variance ratio
explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)

# Determine the number of components to explain at least 40% of the variance
n_components = np.argmax(explained_var >= 0.40) + 1

# Extract the principal components
principal_components = eigenvectors[:, :n_components]

# Create a DataFrame for principal components
pca_component = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)], index=X.columns)

# Plotting heatmap of PCA components
plt.figure(figsize=(8, 6))
sns.heatmap(pca_component, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('PCA Components')
plt.show()

# Matrix multiplication or dot Product
Z_pca = Z @ pca_component
# Rename the columns name
Z_pca.rename({'PC1': 'PCA1', 'PC2': 'PCA2'}, axis=1, inplace=True)
# Print the Pricipal Component values
print(Z_pca)



# Importing PCA from sklearn.decomposition
from sklearn.decomposition import PCA

# Let's say, components = 2
pca_sklearn = PCA(n_components=2)
pca_sklearn.fit(Z)
x_pca = pca_sklearn.transform(Z)

# Create the dataframe
df_pca1 = pd.DataFrame(x_pca,
                       columns=['PC{}'.format(i+1) for i in range(2)],
                       index=X.index)
print(df_pca1)


# giving a larger plot
plt.figure(figsize=(8, 6))

# Scatter plot with color based on 'Completed' column
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=np.where(df['Completed'] == 'Completed', 'blue', 'red'))

# labeling x and y axes
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


# components
# Importing PCA from sklearn.decomposition
from sklearn.decomposition import PCA

# Let's say, components = 2
pca_sklearn = PCA(n_components=2)
pca_sklearn.fit(Z)  # Assuming Z is the standardized data
x_pca = pca_sklearn.transform(Z)

# Accessing the principal components
components = pca_sklearn.components_
print("Principal Components:")
print(components)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Assuming df contains your dataset with features and target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Z, df['Completed'], test_size=0.2, random_state=42)

# Initialize a pipeline with PCA and Gradient Boosting Decision Trees
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the data
    ('pca', PCA(n_components=2)),  # Perform PCA
    ('gbdt', GradientBoostingClassifier())  # Gradient Boosting Decision Trees classifier
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict probabilities on the testing set
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Predict on the testing set
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Check if there are positive samples in the test set
if 'Completed' in y_test.unique():
    # Convert labels to binary format
    y_test_binary = (y_test == 'Completed').astype(int)

    # Predict probabilities for the positive class
    y_proba_positive = pipeline.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_proba_positive)
    roc_auc = auc(fpr, tpr)
    

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("No positive samples in the test set.")