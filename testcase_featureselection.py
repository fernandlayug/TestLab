import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

# Load data from Excel file into a DataFrame
df = pd.read_excel("balanced_data.xlsx")

# Separate features (X) and target variable (y)
X = df.drop(columns=["Completed"])
y = df["Completed"]

# Perform feature selection
selector = SelectKBest(score_func=f_regression, k=2)
X_selected = selector.fit_transform(X, y)

# Get selected feature indices
selected_features = selector.get_support(indices=True)

print("Selected feature indices:", selected_features)
