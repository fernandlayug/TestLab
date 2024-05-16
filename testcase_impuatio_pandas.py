import pandas as pd

# Read data from Excel 
data = pd.read_excel("ALSResponses06.xlsx")

# Identify missing values
missing_values = data.isnull().sum()

# Impute missing values for categorical columns with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)

# Impute missing values for numerical columns with Mean
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    mean_value = data[col].mean()
    data[col].fillna(mean_value, inplace=True)

# Save the imputed data back to an Excel file
data.to_excel("imputed_data.xlsx", index=False)
