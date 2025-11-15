import pandas as pd

# Load the Excel file (male metabolite data sheet)
df_male = pd.read_csv("/Users/vcagbara/Downloads/Combined_Case_and_Control.csv")

# Inspect the original dataframe (optional)
print("Original male data head:")
print(df_male.head())

# Assuming that the first column is a key that should become the index,
# set the first column as the index and then transpose the dataframe.
df_male = df_male.set_index(df_male.columns[0]).transpose()

# Inspect the transposed dataframe (optional)
print("Transposed male data head:")
print(df_male.head())

# Write out the transformed dataframe as a CSV file.
output_csv_path = "/Users/vcagbara/Downloads/mixed_metabolite_data_transposed_975.csv"
df_male.to_csv(output_csv_path)

print(f"Transposed CSV file saved as {output_csv_path}")
