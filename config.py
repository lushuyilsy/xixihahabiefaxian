import torch

# 1. 数据和路径配置
# 假设您的数据结构如下：
# data/
#   images/ (存放所有 .nii.gz)
#   labels.json
DATA_DIR = "data/images/"
LABEL_FILE = "data/labels.json" # 这是一个关键的假设文件，见下文解释

# 2. 标签和类别映射
# 这是项目的核心：将文本标签映射到模型的输出通道
# 假设我们要识别3个区域 + 1个背景
# 背景始终是通道 0
LABEL_MAP = {
    "liver": 1,    # 肝脏
    "kidney": 2,   # 肾脏
    "spleen": 3    # 脾脏
    # ... 根据您的需求添加更多
}
# 反向映射，用于推理
CHANNEL_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
# 模型的输出通道数 (不包括背景)
NUM_CLASSES = len(LABEL_MAP)

# 3. 训练超参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1           # 3D数据非常消耗显存，Batch Size 通常为 1 或 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
VAL_SPLIT = 0.2          # 验证集比例

# 4. 图像预处理参数
# 我们将所有图像和掩码调整到这个固定大小
RESIZE_SHAPE = (128, 128, 128) # (Z, Y, X) 或 (D, H, W)

# 5. 输出
MODEL_SAVE_PATH = "ct_roi_model.pth"