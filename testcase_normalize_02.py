import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Read data from Excel file into a DataFrame
data = pd.read_excel("imputed_ALSdata.xlsx")

# Convert all columns to strings
data = data.astype(str)

# Enforce uniform data types for all columns
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


# Separate numerical and categorical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Define preprocessing steps for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps for both types of features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])


# Apply preprocessing to the data
transformed_data = pipeline.fit_transform(data)

# Convert the transformed data back to DataFrame
transformed_df = pd.DataFrame(transformed_data)

# Save the transformed data to an Excel file
transformed_df.to_excel("normalized_data.xlsx", index=False)
