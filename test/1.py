import pandas as pd
import re
from collections import defaultdict, Counter
import numpy as np

# 读取数据
df = pd.read_excel('C:/Users/DELL/Downloads/new.xlsx')

# 数据基本信息分析
print("=== 数据基本信息 ===")
print(f"总记录数：{len(df)}")
print(f"唯一病人数：{df['A列_姓名'].nunique()}")
print(f"存在重复记录的病人数：{sum(df['A列_姓名'].value_counts() > 1)}")

# 查看检查所见文本示例（前3条）
print("\n=== 检查所见文本示例 ===")
for i in range(3):
    print(f"\n病人{i+1}（{df.iloc[i]['A列_姓名']}）：")
    print(df.iloc[i]['第L列_检查所见'][:200] + "..." if len(str(df.iloc[i]['第L列_检查所见'])) > 200 else df.iloc[i]['第L列_检查所见'])

# 检查缺失值

print(f"\n=== 缺失值统计 ===")
print(f"姓名缺失数：{df['A列_姓名'].isna().sum()}")
print(f"检查所见缺失数：{df['第L列_检查所见'].isna().sum()}")

# 过滤有效数据（去除检查所见为空的记录）
df_valid = df.dropna(subset=['第L列_检查所见']).copy()
print(f"\n有效数据记录数（检查所见非空）：{len(df_valid)}")