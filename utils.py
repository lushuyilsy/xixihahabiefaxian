import numpy as np
import torch
from scipy.ndimage import find_objects


def create_mask_from_bboxes(shape, bboxes_info, label_map):
    """
    根据 Bounding Box 列表创建一个多通道的 3D 掩码。

    Args:
        shape (tuple): 目标掩码的形状 (D, H, W) 或 (Z, Y, X)
        bboxes_info (list): [{"label": "liver", "bbox_xyz": [...]}, ...]
        label_map (dict): {"liver": 1, "kidney": 2, ...}

    Returns:
        torch.Tensor: 形状为 (NumClasses, D, H, W) 的掩码
    """
    num_classes = len(label_map)
    # (D, H, W)
    mask = np.zeros((num_classes, shape[0], shape[1], shape[2]), dtype=np.uint8)

    for roi in bboxes_info:
        label_name = roi["label"]
        if label_name not in label_map:
            continue

        channel_idx = label_map[label_name] - 1  # 映射到 0-indexed

        # 假设 bbox_xyz = [x_min, y_min, z_min, x_max, y_max, z_max]
        # 注意: 数组索引是 (z, y, x)
        x_min, y_min, z_min, x_max, y_max, z_max = [int(p) for p in roi["bbox_xyz"]]

        # 确保边界在图像尺寸内
        z_min = max(0, z_min);
        z_max = min(shape[0], z_max)
        y_min = max(0, y_min);
        y_max = min(shape[1], y_max)
        x_min = max(0, x_min);
        x_max = min(shape[2], x_max)

        # 在对应通道上将 BBX 区域设为 1
        if z_min < z_max and y_min < y_max and x_min < x_max:
            mask[channel_idx, z_min:z_max, y_min:y_max, x_min:x_max] = 1

    return torch.from_numpy(mask)


def convert_mask_to_bbox(mask_tensor, threshold=0.5):
    """
    将单个二值化掩码 (D, H, W) 转换为 Bounding Box。

    Args:
        mask_tensor (torch.Tensor): 形状为 (D, H, W) 的预测掩码
        threshold (float): 二值化阈值

    Returns:
        list: [x_min, y_min, z_min, x_max, y_max, z_max] 或 None
    """
    # 1. 应用 Sigmoid (如果模型没带) 和阈值
    binary_mask = (torch.sigmoid(mask_tensor) > threshold).cpu().numpy().astype(np.uint8)

    # 2. 检查是否有任何前景像素
    if binary_mask.sum() == 0:
        return None

    # 3. 查找对象
    # find_objects 返回一个元组列表，每个元组包含(z_slice, y_slice, x_slice)
    locations = find_objects(binary_mask)
    if not locations:
        return None

    # 假设只有一个连通区域，取第一个
    slices = locations[0]
    z_slice, y_slice, x_slice = slices

    # slices 是 (start, stop)
    z_min, z_max = z_slice.start, z_slice.stop
    y_min, y_max = y_slice.start, y_slice.stop
    x_min, x_max = x_slice.start, x_slice.stop

    # 返回 [x_min, y_min, z_min, x_max, y_max, z_max]
    return [x_min, y_min, z_min, x_max, y_max, z_max]