import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import re


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    model_name = 'bert-base-chinese'
    batch_size = 8
    max_len = 256
    epochs = 3
    learning_rate = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = '提取的姓名与检查结论数据 (1).xlsx - 姓名与检查结论.csv'
    label_column = 'Auto_Organ_Label'
    text_column = 'M列_检查结论'


# ==========================================
# 2. 全身代谢器官精细标注逻辑 (核心升级)
# ==========================================
def auto_label_rule_based(text):
    """
    基于规则的精细化自动标注。
    覆盖全身主要代谢器官及其亚区，用于训练模型识别解剖位置。
    """
    if pd.isna(text):
        return "Unknown"
    text = str(text)

    # --- 1. 肝胆胰脾 (代谢核心) ---
    # 胰腺
    if '胰头' in text or '钩突' in text: return '胰腺-胰头/钩突'
    if '胰体' in text: return '胰腺-胰体'
    if '胰尾' in text: return '胰腺-胰尾'
    if '胰' in text and ('代谢' in text or '占位' in text or '肿块' in text): return '胰腺-部位未明'

    # 肝脏 (细分叶段)
    if '肝右叶' in text: return '肝脏-右叶'
    if '肝左叶' in text: return '肝脏-左叶'
    if '肝尾状叶' in text or '尾状叶' in text: return '肝脏-尾状叶'
    if '肝门' in text: return '肝脏-肝门区'
    if '肝' in text and ('代谢' in text or '占位' in text): return '肝脏-部位未明'

    # 胆囊
    if '胆囊' in text: return '胆囊'

    # --- 2. 上消化道 (食管与胃) ---
    # 食管 (分段)
    if '食管' in text:
        if '上段' in text or '颈段' in text: return '食管-上段'
        if '中段' in text: return '食管-中段'
        if '下段' in text: return '食管-下段'
        if '腹段' in text: return '食管-腹段'
        return '食管-部位未明'

    # 胃部 (细分亚区)
    if '幽门' in text or '胃窦' in text: return '胃部-胃窦/幽门'
    if '胃底' in text or '穹隆' in text: return '胃部-胃底'
    if '贲门' in text or '食管胃交界' in text: return '胃部-贲门'
    if '胃体' in text: return '胃部-胃体'
    if '小弯' in text: return '胃部-小弯侧'
    if '大弯' in text: return '胃部-大弯侧'
    # 如果只提到胃壁增厚代谢高，但未提具体部位，归为模糊类
    if '胃' in text and ('代谢' in text or '增厚' in text): return '胃部-部位未明'

    # --- 3. 下消化道 (肠道) ---
    if '升结肠' in text or '盲肠' in text or '回盲部' in text: return '结肠-右半/升结肠'
    if '横结肠' in text: return '结肠-横结肠'
    if '降结肠' in text: return '结肠-左半/降结肠'
    if '乙状结肠' in text: return '结肠-乙状结肠'
    if '直肠' in text: return '直肠'

    # --- 4. 泌尿生殖系统 ---
    # 前列腺 (左右区分)
    if '前列腺' in text:
        if '左' in text and '右' not in text: return '前列腺-左侧叶'
        if '右' in text and '左' not in text: return '前列腺-右侧叶'
        if '中央沟' in text: return '前列腺-中央区'
        if '外周带' in text: return '前列腺-外周带'
        return '前列腺-部位未明/弥漫性'

    # 肾脏
    if '肾' in text and ('上腺' not in text):  # 排除肾上腺
        if '左' in text: return '肾脏-左侧'
        if '右' in text: return '肾脏-右侧'

    # 子宫/附件
    if '子宫' in text or '宫颈' in text: return '子宫/宫颈'
    if '卵巢' in text or '附件' in text: return '附件区(卵巢/输卵管)'

    # --- 5. 内分泌系统 ---
    # 肾上腺
    if '肾上腺' in text:
        if '左' in text: return '肾上腺-左侧'
        if '右' in text: return '肾上腺-右侧'
        return '肾上腺'

    # 甲状腺
    if '甲状腺' in text:
        if '左' in text: return '甲状腺-左叶'
        if '右' in text: return '甲状腺-右叶'
        if '峡部' in text: return '甲状腺-峡部'
        return '甲状腺'

    # --- 6. 其他常见高代谢区 ---
    # 肺部 (简单分叶，如需分段可继续扩展)
    if '肺' in text:
        if '上叶' in text: return '肺-上叶'
        if '中叶' in text: return '肺-中叶'
        if '下叶' in text: return '肺-下叶'
        if '肺门' in text: return '肺-肺门'

    # 骨骼 (代谢常提示转移)
    if '骨' in text and ('代谢' in text or '破坏' in text):
        if '椎' in text: return '骨骼-脊柱'
        if '肋' in text: return '骨骼-肋骨'
        if '盆' in text or '髂' in text: return '骨骼-骨盆'
        return '骨骼-其他'

    # --- 7. 通用代谢异常 (兜底逻辑) ---
    if '代谢' in text and '增高' in text:
        return '其他代谢异常灶'

    return '非代谢相关/正常'


# ==========================================
# 3. 数据集处理类
# ==========================================
class MedicalReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'report_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==========================================
# 4. 数据加载与处理
# ==========================================
def load_data(config):
    print(f"正在加载数据: {config.data_path}")
    try:
        df = pd.read_csv(config.data_path)
    except Exception as e:
        print(f"读取CSV失败，尝试使用默认编码: {e}")
        df = pd.read_csv(config.data_path, encoding='gbk')

    df = df.dropna(subset=[config.text_column])

    # ------------------------------------------
    # 执行自动标注
    # ------------------------------------------
    print("正在进行全身多器官精细化标注...")
    df[config.label_column] = df[config.text_column].apply(auto_label_rule_based)

    # 打印Top 20分布，检查覆盖情况
    print("\n标注结果分布 (Top 20):")
    print(df[config.label_column].value_counts().head(20))

    # 过滤掉 'Unknown'，保留 '非代谢相关/正常' 作为负样本
    df = df[df[config.label_column] != 'Unknown']

    # 标签编码
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df[config.label_column])

    return df, label_encoder


# ==========================================
# 5. 训练流程
# ==========================================
def train_model(model, data_loader, optimizer, device, dataset_len):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / dataset_len, np.mean(losses)


# ==========================================
# 6. 主程序
# ==========================================
if __name__ == '__main__':
    cfg = Config()

    if not os.path.exists(cfg.data_path):
        print(f"错误：找不到文件 {cfg.data_path}。")
    else:
        # 1. 加载并标注
        df, label_encoder = load_data(cfg)
        num_classes = len(label_encoder.classes_)
        print(f"共识别出 {num_classes} 个解剖亚区类别。")

        # 2. 划分数据集
        df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

        print("正在加载 BERT Tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(cfg.model_name)

        train_dataset = MedicalReportDataset(
            texts=df_train[cfg.text_column].to_numpy(),
            labels=df_train['encoded_label'].to_numpy(),
            tokenizer=tokenizer,
            max_len=cfg.max_len
        )

        val_dataset = MedicalReportDataset(
            texts=df_val[cfg.text_column].to_numpy(),
            labels=df_val['encoded_label'].to_numpy(),
            tokenizer=tokenizer,
            max_len=cfg.max_len
        )

        train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

        print(f"正在加载 BERT 模型 (类别数: {num_classes})...")
        model = BertForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=num_classes
        )
        model = model.to(cfg.device)

        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

        print("\n开始训练...")
        for epoch in range(cfg.epochs):
            print(f'Epoch {epoch + 1}/{cfg.epochs}')
            print('-' * 10)
            train_acc, train_loss = train_model(
                model, train_data_loader, optimizer, cfg.device, len(df_train)
            )
            print(f'训练 Loss: {train_loss:.4f} | 准确率: {train_acc:.4f}')

        print("\n训练完成！")

        # ==========================================
        # 7. 预测演示
        # ==========================================
        print("\n--- 预测演示 (验证集样本) ---")
        if len(df_val) > 0:
            # 随机取几个样本看效果
            sample_indices = np.random.choice(len(df_val), min(3, len(df_val)), replace=False)

            for idx in sample_indices:
                sample_text = df_val.iloc[idx][cfg.text_column]
                true_label_idx = df_val.iloc[idx]['encoded_label']
                true_label = label_encoder.inverse_transform([true_label_idx])[0]

                print(f"\n原文: {sample_text[:60]}...")

                encoded_review = tokenizer.encode_plus(
                    sample_text,
                    max_length=cfg.max_len,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True
                )

                input_ids = encoded_review['input_ids'].to(cfg.device)
                attention_mask = encoded_review['attention_mask'].to(cfg.device)

                with torch.no_grad():
                    output = model(input_ids, attention_mask)
                    _, prediction = torch.max(output.logits, dim=1)

                predicted_label = label_encoder.inverse_transform([prediction.item()])[0]
                print(f"规则标注(参考): {true_label}")
                print(f"模型预测: {predicted_label}")
        else:
            print("验证集为空。")