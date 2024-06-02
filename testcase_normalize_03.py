import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the Excel file into a pandas DataFrame
file_path = 'imputed_ALSdata.xlsx'
sheet_name = 'Sheet1'  # Change this to your sheet name if necessary
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Display the first few rows of the DataFrame
print("Original DataFrame:")
print(df.head())

# Select the categorical columns to encode
categorical_columns = ['FatherJob', 'FatherSalaryRange']  # Replace with your column names

# Ensure all categorical columns are of type string
df[categorical_columns] = df[categorical_columns].astype(str)

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Use sparse_output instead of sparse

# Fit and transform the data
encoded_data = encoder.fit_transform(df[categorical_columns])

# Convert the encoded data back to a DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# Drop the original categorical columns and concatenate the new encoded columns
df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# Display the first few rows of the transformed DataFrame
print("Transformed DataFrame:")
print(df.head())

# Save the transformed DataFrame to a new Excel file
output_file_path = 'path_to_save_transformed_file.xlsx'
df.to_excel(output_file_path, index=False)