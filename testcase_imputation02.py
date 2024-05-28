import pandas as pd

# Read data from Excel 
data = pd.read_excel("ALSDatasets.xlsx")

# Identify missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Impute missing values for categorical columns with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = data[col].mode()[0]  # Get the mode (most frequent value)
    data[col].fillna(mode_value, inplace=True)  # Fill missing values with mode

# Impute missing values for numerical columns with the mean
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputed_rows = pd.DataFrame(columns=data.columns)  # Initialize DataFrame to store imputed rows
for col in numerical_cols:
    mean_value = data[col].mean()  # Calculate the mean value
    # Get rows with missing values in the current column
    missing_rows = data[data[col].isnull()]
    # Fill missing values with mean
    data[col].fillna(mean_value, inplace=True)  
    # Add imputed rows to the DataFrame
    imputed_rows = pd.concat([imputed_rows, missing_rows])

# Save the imputed data back to an Excel file
data.to_excel("imputed_ALSdata.xlsx", index=False)

# Display the rows where imputation occurred
print("Rows where imputation occurred:")
print(imputed_rows)
import pandas as pd

# Read data from Excel 
data = pd.read_excel("ALSDatasets.xlsx")

# Identify missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Impute missing values for categorical columns with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_value = data[col].mode()[0]  # Get the mode (most frequent value)
    data[col].fillna(mode_value, inplace=True)  # Fill missing values with mode

# Impute missing values for numerical columns with the mean
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputed_rows = pd.DataFrame(columns=data.columns)  # Initialize DataFrame to store imputed rows
for col in numerical_cols:
    mean_value = data[col].mean()  # Calculate the mean value
    # Get rows with missing values in the current column
    missing_rows = data[data[col].isnull()]
    # Fill missing values with mean
    data[col].fillna(mean_value, inplace=True)  
    # Add imputed rows to the DataFrame
    imputed_rows = pd.concat([imputed_rows, missing_rows])

# Save the imputed data back to an Excel file
data.to_excel("imputed_ALSdata.xlsx", index=False)

# Display the rows where imputation occurred
print("Rows where imputation occurred:")
print(imputed_rows)

