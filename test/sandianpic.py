import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# 1. 准备数据
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
Ours,0.6352,0.7588,1.851,0.711,2.146,3.3915"""

df = pd.read_csv(io.StringIO(data))

# 2. 计算总体误差 (FNV + FPV)，因为它们都是越低越好，加起来可以代表整体的容积误差
df['Error_Imu'] = df['FNV_Imu'] #+ df['FPV_Imu']
df['Error_UKT'] = df['FNV_UKT'] #+ df['FPV_UKT']

# 3. 创建画布，1行2列
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 调整文字标签的纵向偏移量，防止遮挡点
offset_imu = (df['Error_Imu'].max() - df['Error_Imu'].min()) * 0.04
offset_ukt = (df['Error_UKT'].max() - df['Error_UKT'].min()) * 0.04

# ====================
# 子图 1: Imu Dataset
# ====================
# 画其他方法的基础点
ax1.scatter(df['DSC_Imu'], df['Error_Imu'], color='skyblue', s=120, edgecolor='black', alpha=0.8)

# 单独把 Ours 挑出来画成显眼的大红星
ours_imu = df[df['Method'] == 'Ours'].iloc[0]
ax1.scatter(ours_imu['DSC_Imu'], ours_imu['Error_Imu'], color='red', s=600, marker='*', edgecolor='black', zorder=5, label='Ours')

# 给每个点加上方法名字标签
for i, row in df.iterrows():
    weight = 'bold' if row['Method'] == 'Ours' else 'normal'
    color = 'red' if row['Method'] == 'Ours' else 'black'
    ax1.text(row['DSC_Imu'], row['Error_Imu'] - offset_imu, row['Method'],
             fontsize=9, weight=weight, color=color, ha='center', va='bottom')

ax1.set_xlabel('DSC (Higher is better)', fontsize=12, fontweight='bold')
#ax1.set_ylabel('Total Volume Error: FNV + FPV (Lower is better)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Volume Error: FNV (Lower is better)', fontsize=12, fontweight='bold')
ax1.set_title('Imu Dataset: Accuracy vs. Total Error', fontsize=15, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)

# 重要：反转Y轴，使得顶部代表"更低的误差"（更好的性能），这样"右上角"就是最佳象限
ax1.invert_yaxis()

# 添加"最佳区域"提示框
ax1.text(0.95, 0.95, 'Optimal Zone\n(High DSC, Low Error)', transform=ax1.transAxes,
         fontsize=12, fontweight='bold', color='darkgreen', ha='right', va='top',
         bbox=dict(facecolor='lightgreen', alpha=0.3, edgecolor='green', boxstyle='round,pad=0.5'))


# ====================
# 子图 2: UKT Dataset
# ====================
# 画其他方法的基础点
ax2.scatter(df['DSC_UKT'], df['Error_UKT'], color='steelblue', s=120, edgecolor='black', alpha=0.8)

# 单独把 Ours 挑出来画成显眼的大红星
ours_ukt = df[df['Method'] == 'Ours'].iloc[0]
ax2.scatter(ours_ukt['DSC_UKT'], ours_ukt['Error_UKT'], color='red', s=600, marker='*', edgecolor='black', zorder=5, label='Ours')

# 给每个点加上方法名字标签
for i, row in df.iterrows():
    weight = 'bold' if row['Method'] == 'Ours' else 'normal'
    color = 'red' if row['Method'] == 'Ours' else 'black'
    ax2.text(row['DSC_UKT'], row['Error_UKT'] - offset_ukt, row['Method'],
             fontsize=9, weight=weight, color=color, ha='center', va='bottom')

ax2.set_xlabel('DSC (Higher is better)', fontsize=12, fontweight='bold')
#ax2.set_ylabel('Total Volume Error: FNV + FPV (Lower is better)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Volume Error: FNV (Lower is better)', fontsize=12, fontweight='bold')
ax2.set_title('UKT Dataset: Accuracy vs. Total Error', fontsize=15, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)

# 反转Y轴
ax2.invert_yaxis()

# 添加"最佳区域"提示框
ax2.text(0.95, 0.95, 'Optimal Zone\n(High DSC, Low Error)', transform=ax2.transAxes,
         fontsize=12, fontweight='bold', color='darkgreen', ha='right', va='top',
         bbox=dict(facecolor='lightgreen', alpha=0.3, edgecolor='green', boxstyle='round,pad=0.5'))

# 4. 整理并保存图表
plt.tight_layout()
plt.savefig('combined_scatter.png', dpi=300, bbox_inches='tight') # bbox_inches 保证周围字不被截断
print("图片已保存至当前目录下的 combined_scatter.png")

# 如果您用的是 Jupyter Notebook 等交互环境，也可以取消下面这行的注释来直接显示
# plt.show()