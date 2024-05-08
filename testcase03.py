import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_excel('ALSResponses05.xlsx')
X = df.drop(columns=['Completed'])
y = df['Completed']

# Standardize features
scaler = StandardScaler()
Z = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=4)
Z_pca = pca.fit_transform(Z)

# Plot PCA components
pca_df = pd.DataFrame(Z_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
plt.figure(figsize=(8, 6))
sns.heatmap(pca_df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('PCA Components Correlation')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(Z_pca, y, test_size=0.2, random_state=42)


# Model pipeline
pipeline = Pipeline([
    ('pca', PCA(n_components=4)),
    ('gbdt', GradientBoostingClassifier())
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predictions
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
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
