import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1) Load and rename
df_raw = pd.read_csv('/Users/vcagbara/Downloads/mixed_metabolite_data_transposed_975.csv',
                     header=None, encoding='utf-8')
metabolite_names = df_raw.iloc[0, 2:].astype(str).tolist()
df_raw.columns = ['Group', 'Sample_ID'] + metabolite_names
df = df_raw.drop(0).reset_index(drop=True)

# 2) Lump timepoints into Case vs Control
df['Group'] = df['Group'].str.strip()
df = df[df['Group'].isin(['Case M06','Case M18','Control M06','Control M18'])]
# This is exactly the same grouping as your combined‐VIP code:
# both Case M06 & M18 → “Case”, both Control M06 & M18 → “Control”
df['Binary'] = df['Group'].apply(lambda x: 'Case' if x.startswith('Case') else 'Control')

# 3) Extract X and y
met_cols = [c for c in df.columns if c not in ['Group','Sample_ID','Binary']]
X = df[met_cols].apply(pd.to_numeric, errors='coerce')
y = df['Binary'].map({'Case':0, 'Control':1}).values

# 4) Impute, scale
X_imp = SimpleImputer(strategy='mean').fit_transform(X)
X_scl = StandardScaler().fit_transform(X_imp)
X_scl = np.nan_to_num(X_scl)

# 5) Block weights
vars_ = np.var(X_scl, axis=0, ddof=0)
bw = np.zeros_like(vars_)
mask = vars_ > 1e-15
bw[mask] = 1/np.sqrt(vars_[mask])
X_w = X_scl * bw

# 6a) 1‑component PLS for **combined** VIPs
pls_vip = PLSRegression(n_components=1)
pls_vip.fit(X_w, y)

def calculate_vip(pls, block_weights):
    t = pls.x_scores_
    w = pls.x_weights_
    ssx = np.sum(t**2, axis=0)
    total = np.sum(ssx)
    p = w.shape[0]
    vip = np.zeros(p)
    if total == 0:
        return vip
    for k in range(pls.n_components):
        wk = w[:,k] * block_weights
        vip += (wk**2) * (ssx[k]/total)
    return np.sqrt(p * vip)

vip_scores = calculate_vip(pls_vip, bw)
vip_df = pd.DataFrame({'Metabolite': met_cols, 'VIP': vip_scores})
vip_df.to_csv(
    '/Users/vcagbara/Downloads/combined_vip_scores.csv',
    index=False
)
print("Combined VIP scores saved.")

# 6b) 2‑component PLS for plotting
pls_plot = PLSRegression(n_components=2)
pls_plot.fit(X_w, y)
scores = pls_plot.x_scores_

# 7) 4‑Quadrant Score Plot
fig, ax = plt.subplots(figsize=(8,6))

for grp, color in [('Case','blue'), ('Control','red')]:
    sel = df['Binary']==grp
    ax.scatter(
        scores[sel,0], scores[sel,1],
        label=grp, color=color,
        s=80, alpha=0.8, edgecolor='k'
    )

# draw crosshairs
ax.axhline(0, color='grey', lw=1)
ax.axvline(0, color='grey', lw=1)

# clean up spines
for s in ['top','right']:
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# ticks only on bottom/left
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(direction='out', length=6, width=1, labelsize=10)

ax.set_xlabel('Field', fontsize=12, labelpad=10)
ax.set_ylabel('Lab', fontsize=12, labelpad=10)

# quadrant labels
xlim = ax.get_xlim(); ylim = ax.get_ylim()
ax.text(xlim[1]*0.6, ylim[1]*0.8, 'I', fontsize=14, ha='center')
ax.text(xlim[0]*0.4, ylim[1]*0.8, 'II', fontsize=14, ha='center')
ax.text(xlim[0]*0.4, ylim[0]*0.2, 'III', fontsize=14, ha='center')
ax.text(xlim[1]*0.6, ylim[0]*0.2, 'IV', fontsize=14, ha='center')

# **Show legend** so the user can see which color is Case vs Control
leg = ax.legend(title='Group', loc='upper right', frameon=False, fontsize=11)
leg.set_title('Group')

ax.set_title('Lab vs Field', fontsize=14, pad=15)

plt.tight_layout()
plt.savefig(
    '/Users/vcagbara/Downloads/PLSDA_score_plot_4quadrant_combined_labeled(mixed2025).png',
    dpi=300, bbox_inches='tight'
)
plt.show()
