import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import make_column_selector as selector

# Read data from Excel file
excel_file = "balanced_data_3.xlsx"
data = pd.read_excel(excel_file)

# Define columns
ordinal_columns = ['MotherEducation', 'MotherSalaryRange','FatherEducation','FatherSalaryRange','EducationalAttainment','TimeSpent','DistanceHomeSchool','DaysAvailable','PerformanceScale','TravelTime']  # Replace with your actual ordinal columns
numerical_columns = [col for col in data.columns if col not in ordinal_columns + [data.columns[-1]]]

# Separate features and target variable
features = data.iloc[:, :-1]
target = data.iloc[:, -1]

# Function to convert numpy arrays back to DataFrame
def to_dataframe(X, columns):
    return pd.DataFrame(X, columns=columns)

# Define preprocessor for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('ord', OrdinalEncoder(), ordinal_columns)
    ]
)

# Handle missing values, convert to DataFrame, and normalize data
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('to_dataframe', FunctionTransformer(to_dataframe, kw_args={'columns': features.columns})),
    ('preprocessor', preprocessor)
])

# Fit and transform the features data
normalized_features = pipeline.fit_transform(features)

# Convert normalized features back to a DataFrame
normalized_data = pd.DataFrame(normalized_features, columns=numerical_columns + ordinal_columns)

# Save the normalized data to an Excel file
output_excel_file = "normalized_data_1.xlsx"
normalized_data_with_target = pd.concat([normalized_data, target.reset_index(drop=True)], axis=1)
normalized_data_with_target.to_excel(output_excel_file, index=False)

# Visualization before normalization
plt.figure(figsize=(12, 6))
for i, column in enumerate(features.columns):
    plt.subplot(2, len(features.columns), i + 1)
    sns.histplot(data[column], kde=True)
    plt.title("Before Normalization")
    plt.xlabel("")
    plt.ylabel("")

# Visualization after normalization
plt.figure(figsize=(12, 6))
for i, column in enumerate(normalized_data.columns):
    plt.subplot(2, len(features.columns), len(features.columns) + i + 1)
    sns.histplot(data=normalized_data, x=column, kde=True)
    plt.title("After Normalization")
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.show()
