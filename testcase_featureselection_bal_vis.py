import pandas as pd
import matplotlib.pyplot as plt
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
    return X_selected, selected_features, selector

# Specify the number of features you want to select (k)
k_features = 20

# Perform feature selection
X_selected, selected_features, selector = select_features(X, y, k=k_features)

# Visualize selected features
plt.figure(figsize=(10, 6))
plt.barh(selected_features, selector.scores_[selector.get_support()], color='skyblue')
plt.xlabel('ANOVA F-value')
plt.title('Top {} Selected Features'.format(k_features))
plt.gca().invert_yaxis()  # Invert y-axis to display features from top to bottom
plt.show()
