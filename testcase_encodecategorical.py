import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Step 1: Read the Excel dataset
excel_file = 'imputed_ALSdata.xlsx'
df = pd.read_excel(excel_file)

# Step 2: Encode categorical features
# Assuming 'categorical_cols' is a list containing the column names of categorical features
categorical_cols = ['Gender', 'CivilStatus', 'Home Address Type']
# Initialize one-hot encoder
encoder = OneHotEncoder()
# Fit and transform the categorical columns
encoded_cols = encoder.fit_transform(df[categorical_cols])
# Create a DataFrame for the encoded categorical columns
encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
# Concatenate the original DataFrame with the encoded categorical columns
df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# Step 3: Save the encoded dataset to Excel
output_file = 'encoded_data.xlsx'
df_encoded.to_excel(output_file, index=False)