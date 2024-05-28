import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load the Excel file
file_path = 'ALSDatasets.xlsx'
sheet_name = 'ALSsheet'  # Change to your sheet name if necessary

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Ensure uniform data types
df = df.applymap(lambda x: str(x) if isinstance(x, (int, float)) and not pd.isnull(x) else x)

# Display the first few rows of the DataFrame
print("Original DataFrame with Missing Values:")
print(df.head())

# Separate the columns into numeric and non-numeric
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
non_numeric_features = df.select_dtypes(include=['object']).columns

# Create an imputer for numeric data
numeric_transformer = SimpleImputer(strategy='mean')

# Create an imputer for non-numeric data
non_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Ensure the output is dense (not sparse)
])

# Combine the numeric and non-numeric transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', non_numeric_transformer, non_numeric_features)
    ])

# Fit and transform the data
df_imputed = preprocessor.fit_transform(df)

# Extract feature names after OneHotEncoding
non_numeric_transformed = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(non_numeric_features)
all_features = list(numeric_features) + list(non_numeric_transformed)

# Convert the result back into a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=all_features)

# Display the DataFrame after imputation
print("\nDataFrame after Imputation:")
print(df_imputed.head())

# Save the imputed DataFrame to a new Excel file
output_file_path = 'Imputed_ALSDataset.xlsx'
df_imputed.to_excel(output_file_path, index=False)

print(f"\nImputed data saved to {output_file_path}")
