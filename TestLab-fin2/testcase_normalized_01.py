import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Read data from Excel file
excel_file = "balanced_data_3.xlsx"
data = pd.read_excel(excel_file)

# Define columns
ordinal_columns = ['MotherEducation', 'MotherSalaryRange','FatherEducation','FatherSalaryRange','EducationalAttainment','TimeSpent','DistanceHomeSchool','DaysAvailable','PerformanceScale']  # Replace with your actual ordinal columns
numerical_columns = [col for col in data.columns if col not in ordinal_columns  + [data.columns[-1]]]

# Separate features and target variable
features = data.iloc[:, :-1]
target = data.iloc[:, -1]

# Define preprocessor for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('ord', OrdinalEncoder(), ordinal_columns),
     
    ])

# Handle missing values and normalize data
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('preprocessor', preprocessor)
])

# Fit and transform the features data
normalized_features = pipeline.fit_transform(features)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(normalized_features, target)

# Save the balanced data to an Excel file
balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=pipeline.named_steps['preprocessor'].get_feature_names_out()), pd.Series(y_resampled, name=target.name)], axis=1)
output_excel_file = "balanced_data_normalized_1.xlsx"
balanced_data.to_excel(output_excel_file, index=False)

# Visualization before normalization
plt.figure(figsize=(12, 6))
for i, column in enumerate(features.columns):
    plt.subplot(2, len(features.columns), i+1)
    sns.histplot(data[column], kde=True)
    plt.title("Before Normalization")
    plt.xlabel("")
    plt.ylabel("")

# Visualization after normalization and SMOTE
for i, column in enumerate(balanced_data.columns[:-1]):  # excluding the target column
    plt.subplot(2, len(features.columns), len(features.columns)+i+1)
    sns.histplot(data=balanced_data, x=column, kde=True)
    plt.title("After Normalization and SMOTE")
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.show()
