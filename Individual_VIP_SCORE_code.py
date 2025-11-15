import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

################################
# 1) Read the CSV with NO header
################################
df_raw = pd.read_csv("/Users/vcagbara/Downloads/mixed_metabolite_data_transposed_975.csv",
                     encoding='utf-8', header=None)
print("Initial shape:", df_raw.shape)
# In your file, row 0 (the first row) holds the metabolite names starting from column index 2 onward.

################################
# 2) Extract metabolite names from row 0, starting at column 2 (include the first metabolite)
################################
metabolite_names_str = df_raw.iloc[0, 2:].astype(str).tolist()
print("Metabolite names from first row, starting from 3rd column (total={}):".format(len(metabolite_names_str)))
print(metabolite_names_str)

################################
# 3) Rename the first 2 columns to 'Group' and 'Sample_ID'
################################
df_raw.rename(columns={
    0: 'Group',
    1: 'Sample_ID'
}, inplace=True)

################################
# 4) Rename columns 2+ with the extracted metabolite names
################################
all_columns = df_raw.columns.tolist()
# Starting from column index 2, assign the extracted metabolite names.
for i, col_index in enumerate(range(2, df_raw.shape[1])):
    new_name = metabolite_names_str[i]  # Map the i-th metabolite name
    all_columns[col_index] = new_name
df_raw.columns = all_columns
print("\nColumns after renaming:\n", df_raw.columns)

################################
# 5) Drop the first row (it contained the metabolite names)
################################
df_raw = df_raw.drop(index=0).reset_index(drop=True)
print("Shape after dropping row 0:", df_raw.shape)

################################
# 6) Ensure the Group values are properly stripped and filter to only 'Case BL' and 'Control BL'
################################
df_raw['Group'] = df_raw['Group'].astype(str).str.strip()
valid_groups = ['Case M06', 'Case M18', 'Control M06', 'Control M18']
df_raw = df_raw[df_raw['Group'].isin(valid_groups)].copy()

# Map group labels
group_map = {
    'Case M06': 'Case M06=1',
    'Case M18': 'Case M18=2',
    'Control M06': 'Control M06=3',
    'Control M18': 'Control M18=4'
}
df_raw['MultiClassLabel'] = df_raw['Group'].map(group_map)
y_multi = df_raw['MultiClassLabel'].values
print("Sample counts per group:\n", df_raw['MultiClassLabel'].value_counts())

################################
# 7) Identify metabolite columns (keep all columns except non-metabolite ones)
################################
non_metabolite_cols = ['Group', 'Sample_ID', 'MultiClassLabel']
metabolite_cols = [c for c in df_raw.columns if c not in non_metabolite_cols]
print("\nMetabolite columns identified (total={}):".format(len(metabolite_cols)))
print(metabolite_cols)

# Convert metabolite data to numeric
X_subset = df_raw[metabolite_cols].apply(pd.to_numeric, errors='coerce')
print("X_subset shape:", X_subset.shape)

################################
# 8) Impute missing values (keep all columns)
################################
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_subset)

################################
# 9) Standard Scaling
################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

################################
# 10) Compute block weights (1/sqrt(variance))
################################
variances = np.var(X_scaled, axis=0, ddof=0)
block_weights = np.zeros_like(variances, dtype=float)
non_zero_mask = (variances > 1e-15)
block_weights[non_zero_mask] = 1 / np.sqrt(variances[non_zero_mask])
block_weights[np.isinf(block_weights)] = 0
X_weighted = X_scaled * block_weights

################################
# 11) Encode Groups into numeric values
################################
ordered_labels = ['Case M06=1', 'Case M18=2', 'Control M06=3', 'Control M18=4']
label_to_int = {label: idx for idx, label in enumerate(ordered_labels)}
y_numeric = np.array([label_to_int[val] for val in y_multi])

################################
# 12) Fit PLS-DA with 1 component
################################
pls = PLSRegression(n_components=1)
pls.fit(X_weighted, y_numeric)
print("PLS x_weights_ shape:", pls.x_weights_.shape)

################################
# 13) Define VIP function (with block weighting)
################################
def calculate_vip(pls, X, block_weights):
    p = X.shape[1]
    t = pls.x_scores_
    w = pls.x_weights_
    ssx = np.sum(t**2, axis=0)
    total_ssx = np.sum(ssx)
    if total_ssx == 0:
        print("Error: PLS model explains no variance. VIP=0.")
        return np.zeros(p)
    vip = np.zeros(p)
    for k in range(pls.n_components):
        w_k = w[:, k] * block_weights
        vip += (w_k**2) * (ssx[k] / total_ssx)
    return np.sqrt(p * vip)

################################
# 14) Calculate VIP scores
################################
vip_scores = calculate_vip(pls, X_weighted, block_weights)
print("Number of VIP scores computed:", len(vip_scores))
print("Number of metabolite names:", len(metabolite_cols))

################################
# 15) Build DataFrame (keeping original order)
################################
vip_df = pd.DataFrame({
    'Metabolite': metabolite_cols,  # Use all metabolite columns in original order
    'VIP': vip_scores
})

# Do not sort so as to preserve the original order
print("\nFirst 10 Metabolites in Original Order:")
print(vip_df.head(10))

################################
# 16) Save the VIP scores to CSV
################################
output_path = '/Users/vcagbara/Downloads/paired_vip_scores_block_weighted(indiv)10.csv'
vip_df.to_csv(output_path, index=False)
print(f"VIP scores saved to {output_path}")
