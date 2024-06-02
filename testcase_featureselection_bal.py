import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Load data from Excel file
df = pd.read_excel('balanced_data.xlsx')

# Split data into features (X) and target variable (y)
X = df.drop(columns=['Completed'])
y = df['Completed']

# Feature selection using SelectKBest with ANOVA F-value scoring
def select_features(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features

# Specify the number of features you want to select (k)
k_features = 5

# Perform feature selection
X_selected, selected_features = select_features(X, y, k=k_features)

print("Selected Features:", selected_features)
print("Shape of X_selected:", X_selected.shape)
