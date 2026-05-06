import torch
import nibabel as nib
import numpy as np
import config
from model import get_model
from utils import convert_mask_to_bbox
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    NormalizeIntensity,
    Resize,
    EnsureType
)


# 1. 加载模型 (用于推理)
def load_inference_model(model_path):
    print("加载模型...")
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    print("模型加载完毕。")
    return model


# 2. 定义推理预处理流程
# 必须与训练时的数据处理 *完全* 一致
def get_inference_transforms():
    return Compose([
        LoadImage(image_only=True),  # 加载 NIfTI
        EnsureChannelFirst(),  # (D, H, W) -> (1, D, H, W)
        Resize(spatial_size=config.RESIZE_SHAPE),
        NormalizeIntensity(),
        EnsureType(dtype=torch.float32)
    ])


# 3. 核心功能函数
def get_roi_bbox_from_text(model, image_path, text_label):
    """
    输入CT图像路径和文本标签，输出对应的 Bounding Box。

    Args:
        model: 已加载的训练模型
        image_path (str): .nii.gz 文件路径
        text_label (str): 文本标签 (例如 "liver" 或 "kidney")

    Returns:
        list: [x_min, y_min, z_min, x_max, y_max, z_max] 或 None
    """

    print(f"开始推理... 图像: {image_path}, 目标: '{text_label}'")

    # 1. 检查文本标签是否在我们的类别中
    if text_label not in config.LABEL_MAP:
        print(f"[错误] 文本标签 '{text_label}' 未在 config.LABEL_MAP 中定义。")
        return None

    # 2. 获取对应的模型输出通道索引
    # 记住: "liver" 映射到 1，在 0-indexed 的掩码中是通道 0
    channel_index = config.LABEL_MAP[text_label] - 1

    # 3. 加载和预处理图像
    try:
        transforms = get_inference_transforms()
        image_tensor = transforms(image_path)  # (1, D, H, W)

        # 添加 Batch 维度 (B, C, D, H, W)
        image_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)  # (1, 1, D, H, W)

    except Exception as e:
        print(f"[错误] 加载或处理图像失败: {e}")
        return None

    # 4. 模型推理
    with torch.no_grad():
        # (1, NumClasses, D, H, W)
        pred_masks = model(image_tensor)

    # 5. 提取我们感兴趣的特定掩码
    # (1, D, H, W) -> (D, H, W)
    target_mask = pred_masks[0, channel_index, :, :, :]

    # 6. 将掩码转换为 Bounding Box
    # utils.py 中的函数会处理 Sigmoid、阈值和坐标提取
    bbox = convert_mask_to_bbox(target_mask, threshold=0.5)

    if bbox:
        print(f"成功！在 {config.RESIZE_SHAPE} 尺度下找到 Bounding Box: {bbox}")
    else:
        print(f"未能在图像中找到 '{text_label}' (阈值=0.5)。")

    # 注意: 这个 BBX 是相对于 *Resize后* 的图像 (128, 128, 128)
    # 如果需要映射回原始坐标，需要保存 仿射矩阵 (Affine) 并进行逆变换
    # (这会使代码复杂度增加很多，暂时我们先返回 resize 后的坐标)

    return bbox


# --- 主程序：演示如何使用 ---
if __name__ == "__main__":
    # 1. 确保模型已训练并存在
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"[错误] 模型文件 {config.MODEL_SAVE_PATH} 未找到。")
        print("请先运行 train.py 进行训练。")
    else:
        # 2. 加载模型
        model = load_inference_model(config.MODEL_SAVE_PATH)

        # 3. 准备一个测试图像
        # 我们使用训练集的第一张图作为演示
        try:
            with open(config.LABEL_FILE, 'r') as f:
                all_data_info = json.load(f)

            test_item = all_data_info[0]
            test_image_path = os.path.join(config.DATA_DIR, test_item["image_id"])

            # 4. 演示推理

            # 示例 1: 查找 "liver"
            text_query_1 = "liver"
            bbox_1 = get_roi_bbox_from_text(model, test_image_path, text_query_1)

            # 示例 2: 查找 "kidney"
            text_query_2 = "kidney"
            bbox_2 = get_roi_bbox_from_text(model, test_image_path, text_query_2)

            # 示例 3: 查找一个不存在的标签
            text_query_3 = "brain"
            bbox_3 = get_roi_bbox_from_text(model, test_image_path, text_query_3)

        except FileNotFoundError:
            print("[错误] 演示失败。")
            print("请确保 'data/images' 和 'data/labels.json' 存在，并且 train.py 已运行。")
        except Exception as e:
            print(f"[错误] 演示过程中发生意外: {e}")