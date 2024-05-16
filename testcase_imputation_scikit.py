import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Read data from Excel
data = pd.read_excel("ALSResponses06.xlsx")

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Imputation strategies for numerical and categorical data
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Define preprocessing steps for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numerical_cols),
        ('cat', categorical_imputer, categorical_cols)
    ])

# Define the pipeline including preprocessing
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the data
imputed_data = pipeline.fit_transform(data)

# Convert the transformed data back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols.tolist() + categorical_cols.tolist())

# Save the imputed data back to an Excel
imputed_df.to_excel("imputed_data.xlsx", index=False)


