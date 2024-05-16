import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load data from Excel file into a DataFrame
df = pd.read_excel("ALSResponses05.xlsx")

# Separate features (X) and target variable (y)
X = df.drop(columns=["Completed"])
y = df["Completed"]

# Perform feature selection with SelectKBest and chi2
selector = SelectKBest(score_func=chi2, k=2)
X_selected = selector.fit_transform(X, y)

# Get selected feature indices
selected_features = selector.get_support(indices=True)

# Use the selected features for cross-validation
clf = DecisionTreeClassifier()  # Example classifier, replace with your choice
scores = cross_val_score(clf, X_selected, y, cv=5)  # 5-fold cross-validation
mean_accuracy = scores.mean()

print("Selected feature indices:", selected_features)
print("Mean cross-validation accuracy:", mean_accuracy)
