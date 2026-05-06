import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

data = """Method,DSC_Imu,DSC_UKT,FNV_Imu,FNV_UKT,FPV_Imu,FPV_UKT
nnU-Net,0.5082,0.6877,14.1501,1.726,11.8298,6.7098
Max.sh,0.4075,0.7546,30.3385,0.7135,38.8658,3.9097
HKURad,0.6121,0.6993,10.9226,3.3913,4.8894,2.7016
WukongRT,0.6269,0.7276,9.4248,1.1505,6.7777,3.5987
BAMF AI,0.6731,0.7028,11.4233,1.4985,3.251,1.7582
zero_sugar,0.6645,0.7428,12.3798,4.4353,0.6367,0.8774
Shadab,0.6224,0.7325,11.529,0.8563,2.1661,1.9811
QuantIF,0.5865,0.7704,6.7847,2.3469,10.103,0.7639
Lennonlychan,0.6351,0.7509,12.6545,1.238,1.9545,1.2944
Airamatrix,0.6318,0.7611,12.5692,1.667,1.9341,0.9169
UIH_CRI_SIL,0.6259,0.7977,4.731,1.0125,2.0065,2.3335
StockholmTrio,0.5836,0.7739,8.5486,1.2404,2.0091,1.8555
HussainAlasmawi,0.6435,0.7857,2.5107,0.693,3.4181,2.9321
IKIM,0.6108,0.7993,2.7814,1.0987,4.9938,2.7151
LesionTracer,0.6684,0.7757,1.8375,0.4875,1.7595,3.0754
Ours,0.6052,0.6988,1.851,0.711,2.146,3.3915"""

df = pd.read_csv(io.StringIO(data))

fig, axes = plt.subplots(3, 1, figsize=(12, 16))
plt.subplots_adjust(hspace=0.5)

metrics = [
    ('DSC', 'DSC_Imu', 'DSC_UKT', 'DSC (Higher is better)'),
    ('FNV', 'FNV_Imu', 'FNV_UKT', 'FNV (Lower is better)'),
    ('FPV', 'FPV_Imu', 'FPV_UKT', 'FPV (Lower is better)')
]

x = np.arange(len(df['Method']))
width = 0.35

for i, (name, col_imu, col_ukt, title) in enumerate(metrics):
    ax = axes[i]

    colors_imu = ['red' if m == 'Ours' else 'skyblue' for m in df['Method']]
    colors_ukt = ['darkred' if m == 'Ours' else 'steelblue' for m in df['Method']]

    ax.bar(x - width / 2, df[col_imu], width, label='Imu', color=colors_imu, edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, df[col_ukt], width, label='UKT', color=colors_ukt, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Value')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Method'], rotation=45, ha='right')

    import matplotlib.patches as mpatches

    imu_patch = mpatches.Patch(color='skyblue', label='Imu (Others)')
    ukt_patch = mpatches.Patch(color='steelblue', label='UKT (Others)')
    ours_imu_patch = mpatches.Patch(color='red', label='Imu (Ours)')
    ours_ukt_patch = mpatches.Patch(color='darkred', label='UKT (Ours)')
    ax.legend(handles=[imu_patch, ukt_patch, ours_imu_patch, ours_ukt_patch], loc='best')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)
print("图片已成功保存为 performance_comparison.png")