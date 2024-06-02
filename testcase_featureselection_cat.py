import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# Load data from Excel file into a DataFrame
df = pd.read_excel("balanced_data.xlsx")

# Separate features (X) and target variable (y)
X = df.drop(columns=["Completed"])
y = df["Completed"]

# Perform feature selection
selector = SelectKBest(score_func=chi2, k=2)  # Use chi2 as scoring function
X_selected = selector.fit_transform(X, y)

# Get selected feature indices
selected_features = selector.get_support(indices=True)

print("Selected feature indices:", selected_features)
