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

################################
# 3) Rename the first 2 columns to 'Group' and 'Sample_ID'
################################
df_raw.rename(columns={0: 'Group', 1: 'Sample_ID'}, inplace=True)

################################
# 4) Rename columns 2+ with the extracted metabolite names
################################
all_columns = df_raw.columns.tolist()
for i, col_index in enumerate(range(2, df_raw.shape[1])):
    all_columns[col_index] = metabolite_names_str[i]
df_raw.columns = all_columns

################################
# 5) Drop the first row (it contained the metabolite names)
################################
df_raw = df_raw.drop(index=0).reset_index(drop=True)

################################
# 6) Filter to only the four time‐point groups
################################
df_raw['Group'] = df_raw['Group'].str.strip()
valid_groups = ['Case M06','Case M18','Control M06','Control M18']
df_raw = df_raw[df_raw['Group'].isin(valid_groups)].copy()

# Map group labels
group_map = {
    'Case M06': 'Case=1',
    'Case M18': 'Case=1',
    'Control M06': 'Control=2',
    'Control M18': 'Control=2'
}
df_raw['MultiClassLabel'] = df_raw['Group'].map(group_map)
y_multi = df_raw['MultiClassLabel'].values

################################
# 7) Identify metabolite columns
################################
non_met_cols = ['Group','Sample_ID','MultiClassLabel']
metabolite_cols = [c for c in df_raw.columns if c not in non_met_cols]
X_subset = df_raw[metabolite_cols].apply(pd.to_numeric, errors='coerce')

################################
# 8) Impute missing values
################################
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_subset)

################################
# 9) Standard Scaling
################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# <<< UPDATED LINE >>> 
# Replace any NaN or infinite values with 1/10th instead of 0
X_scaled = np.nan_to_num(X_scaled, nan=0.1, posinf=0.1, neginf=0.1)

################################
# 10) Compute block weights (1/sqrt(variance))
################################
variances = np.var(X_scaled, axis=0, ddof=0)
block_weights = np.zeros_like(variances, dtype=float)
non_zero_mask = variances > 1e-15
block_weights[non_zero_mask] = 1 / np.sqrt(variances[non_zero_mask])
block_weights[np.isinf(block_weights)] = 0
X_weighted = X_scaled * block_weights

################################
# 11) Encode Groups into numeric values
################################
ordered_labels = ['Case=1','Control=2']
label_to_int = {lbl: idx for idx, lbl in enumerate(ordered_labels)}
y_numeric = np.array([label_to_int[val] for val in y_multi])

################################
# 12) Fit PLS-DA with 1 component
################################
pls = PLSRegression(n_components=1)
pls.fit(X_weighted, y_numeric)

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

################################
# 15) Build and save DataFrame
################################
vip_df = pd.DataFrame({
    'Metabolite': metabolite_cols,
    'VIP': vip_scores
})
vip_df.to_csv('/Users/vcagbara/Downloads/paired_vip_scores_block_weighted(mixed)(2025)3.csv', index=False)
print("VIP scores saved.")
