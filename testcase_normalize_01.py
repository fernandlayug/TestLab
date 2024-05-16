import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Read data from Excel file into a DataFrame
data = pd.read_excel("ALSResponses06.xlsx")

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Define preprocessing steps for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the pipeline including preprocessing and optional normalization
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('scaler', StandardScaler())])

# Fit and transform the data
normalized_data = pipeline.fit_transform(data)

# Convert the normalized data back to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=numerical_cols.tolist() +
                             pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out().tolist())

# Save the normalized data back to an Excel file if needed
normalized_df.to_excel("normalized_data.xlsx", index=False)
