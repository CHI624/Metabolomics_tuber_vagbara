import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ————— 1) Load & rename —————
df_raw = pd.read_csv(
    '/Users/vcagbara/Downloads/mixed_by_gender_corr_metabolite_data_transposed_974.csv',
    header=None, encoding='utf-8'
)
# first row holds metabolite names from col 2 onward
metabolite_names = df_raw.iloc[0, 2:].astype(str).tolist()
df_raw.columns = ['Group', 'Sample_ID'] + metabolite_names

# drop that header row
df = df_raw.drop(0).reset_index(drop=True)

# ————— 2) Filter to only the four time‑point groups —————
df['Group'] = df['Group'].str.strip()
valid = ['Case M06','Case M18','Control M06','Control M18']
df = df[df['Group'].isin(valid)].copy()

# map each to a numeric label 0–3 for PLS‑DA
label_map = {'Case M06':1,'Case M18':1,'Control M06':2,'Control M18':2}
df['y_multi'] = df['Group'].map(label_map)

# ————— 3) Build X matrix & y vector —————
met_cols = [c for c in df.columns if c not in ['Group','Sample_ID','y_multi']]
X = df[met_cols].apply(pd.to_numeric, errors='coerce')
y = df['y_multi'].values

# ————— 4) Impute & Standardize —————
X_imp = SimpleImputer(strategy='mean').fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imp)

# replace any NaN or inf with 0.1 instead of 0
X_scaled = np.nan_to_num(X_scaled, nan=0.1, posinf=0.1, neginf=0.1)

# ————— 5) Block‑weighting (1/sqrt(var)) —————
vars_ = np.var(X_scaled, axis=0, ddof=0)
bw = np.zeros_like(vars_)
mask = vars_ > 1e-15
bw[mask] = 1/np.sqrt(vars_[mask])
X_weighted = X_scaled * bw

# ————— 6) Fit PLS‑DA (2 components) —————
pls = PLSRegression(n_components=1)
pls.fit(X_weighted, y)
scores = pls.x_scores_

# ————— 7) 4‑Quadrant Score Plot with labels —————
fig, ax = plt.subplots(figsize=(8,6))

# define colors & markers per group
style_map = {
    'Case M06':    ('#1f77b4','o'),
    'Case M18':    ('#17becf','s'),
    'Control M06': ('#d62728','^'),
    'Control M18': ('#9467bd','D'),
}

# scatter each group
for grp, (col, mkr) in style_map.items():
    sel = df['Group'] == grp
    ax.scatter(
        scores[sel,0], scores[sel,1],
        label=grp,
        color=col,
        marker=mkr,
        s=80,
        edgecolor='none',
        alpha=0.8
    )

# annotate each point with its Sample_ID
##for i, sid in enumerate(df['Sample_ID']):
  ##  x, y_ = scores[i,0], scores[i,1]
    ##ax.text(x + 0.02, y_ + 0.02, sid, fontsize=8, color='black', alpha=0.7)

# draw axes through the origin
ax.axhline(0, color='grey', lw=1)
ax.axvline(0, color='grey', lw=1)

# move bottom/left spines to zero
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

# ticks only on bottom/left, pointing out
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(direction='out', length=6, width=1, labelsize=10)

# axis labels and quadrant letters
ax.set_xlabel('PLS Component 1', fontsize=12, labelpad=10)
ax.set_ylabel('PLS Component 2', fontsize=12, labelpad=10)
xlim, ylim = ax.get_xlim(), ax.get_ylim()
ax.text(xlim[1]*0.6, ylim[1]*0.8, 'I',  fontsize=14, ha='center')
ax.text(xlim[0]*0.4, ylim[1]*0.8, 'II', fontsize=14, ha='center')
ax.text(xlim[0]*0.4, ylim[0]*0.2, 'III',fontsize=14, ha='center')
ax.text(xlim[1]*0.6, ylim[0]*0.2, 'IV', fontsize=14, ha='center')

# title & legend
ax.set_title('PLS‑DA 4‑Group Score Plot (Individual Preprocessing)', fontsize=14, pad=15)
ax.legend(title='Group', frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig(
    '/Users/vcagbara/Downloads/PLSDA_score_plot_mixed_gender1_transposed.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
