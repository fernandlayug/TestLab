import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

# Load data from Excel file
df = pd.read_excel('balanced_data_1.xlsx')

# Split data into features (X) and target variable (y)
X = df.drop(columns=['Completed'])
y = df['Completed']

# Handle missing values by imputing with the mean (can be adjusted to other strategies if needed)
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature Engineering
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Create new column names for polynomial features
poly_feature_names = [f'poly_{i}' for i in range(X_poly.shape[1])]
X = pd.DataFrame(X_poly, columns=poly_feature_names)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Oversampling using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Function to perform feature selection with cross-validation
def select_features_cv(X, y, k=20, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    scores = np.zeros(X.shape[1])
    fold_scores = []  # List to store scores of each fold

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        selector = SelectKBest(score_func=mutual_info_classif, k='all')
        selector.fit(X_train, y_train)
        fold_scores.append(selector.scores_)
        scores += selector.scores_

    scores /= n_splits
    selector.scores_ = scores
    selected_features_indices = np.argsort(scores)[-k:]
    selected_features = X.columns[selected_features_indices]
    
    return X.iloc[:, selected_features_indices], selected_features, selector, fold_scores

# Specify the number of features you want to select (k)
k_features = 20

# Perform feature selection with cross-validation
X_selected, selected_features, selector, fold_scores = select_features_cv(X_resampled, y_resampled, k=k_features)

# Visualizations...

# Distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(y_resampled)
plt.title('Distribution of Target Variable (After Resampling)')
plt.xlabel('Completed')
plt.ylabel('Count')
plt.show()

# Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=selector.scores_, y=X_selected.columns)
plt.title('Feature Importance (Mutual Information)')
plt.xlabel('Mutual Information Score')
plt.ylabel('Features')
plt.show()

# Find the best regularization parameter using LassoCV
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_selected, y_resampled)

# Apply Lasso Regression with the best alpha
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_selected, y_resampled)

# Cross-validation with Logistic Regression
logreg = LogisticRegression(max_iter=1000)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Regularization parameter
grid_search = GridSearchCV(logreg, param_grid, cv=5)
grid_search.fit(X_selected, y_resampled)
best_logreg = grid_search.best_estimator_

# Evaluate the model
cv_scores = cross_val_score(best_logreg, X_selected, y_resampled, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_selected, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train and evaluate Logistic Regression on validation set
best_logreg.fit(X_train, y_train)
train_score = best_logreg.score(X_train, y_train)
val_score = best_logreg.score(X_val, y_val)

print("Training set score:", train_score)
print("Validation set score:", val_score)
