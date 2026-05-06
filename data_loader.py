import os
import json
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Resized,
    EnsureTyped
)
from monai.data import decollate_batch
from monai.utils import first

import config
from utils import create_mask_from_bboxes


class CTScanDataset(Dataset):
    def __init__(self, data_list, data_dir, transforms):
        self.data_list = data_list  # data_list 是从 labels.json 来的列表
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_name = item["image_id"]
        image_path = os.path.join(self.data_dir, image_name)

        # 1. 创建数据字典 (MONAI 推荐)
        data_dict = {"image": image_path}

        # 2. 应用基础变换 (加载、调整通道、调整大小、归一化)
        # 注意: 我们稍后手动添加掩码
        data_dict = self.transforms(data_dict)

        # 3. 创建目标掩码 (Target Mask)
        # 获取调整大小后的图像形状 (D, H, W)
        # (我们假设 Resized 变换是最后一个)
        img_shape_dhw = data_dict["image"].shape[1:]  # 形状是 [C, D, H, W]

        # 4. 从 BBX 创建掩码
        # 注意: create_mask_from_bboxes 需要原始的 bbx 坐标
        # MONAI 的 Resized 会自动调整 仿射矩阵 (affine)
        # 一个更鲁棒的方法是使用 MONAI 的 BoundingBox 变换，
        # 但为了简化，我们假设 BBX 坐标是基于原始图像的，
        # 并且 create_mask_from_bboxes 将在 *原始* 尺度上创建掩码
        # 然后 Resized 变换也应该应用于掩码。

        # *** 简化的方法 (如下) ***
        # 假设 `labels.json` 中的 bbx 坐标是 *相对于原始图像* 的。
        # 我们需要在加载后、Resize 之前创建掩码。

        # ... (
        #     为了简化这个示例项目，我们将采取一个捷径：
        #     我们假设 Resized 变换也会应用到我们生成的掩码上。
        #     一个更正确的流程会更复杂，涉及仿射变换。
        # )

        # 让我们重新定义流程，使其更正确且简单

        # 目标：我们需要图像和掩码都被 Resize 到 (128, 128, 128)

        # 1. 加载图像 (只加载，不变换)
        loader = LoadImaged(keys=["image"])
        loaded_data = loader({"image": image_path})

        # 2. 获取原始图像形状 (D, H, W)
        original_shape = loaded_data["image"].shape[1:]  # MONAI 加载后是 (C, D, H, W)

        # 3. 在原始尺度上创建掩码
        # (NumClasses, D, H, W)
        mask = create_mask_from_bboxes(
            shape=original_shape,
            bboxes_info=item["rois"],
            label_map=config.LABEL_MAP
        )

        # 4. 将图像和掩码放入数据字典
        data_dict = {
            "image": loaded_data["image"],  # (1, D, H, W)
            "label": mask  # (NumClasses, D, H, W)
        }

        # 5. 应用所有变换 (Resize, Normalize等)
        # 我们需要一个新的变换流程

        final_transforms = Compose([
            # 已经加载过了，所以跳过 LoadImaged
            EnsureChannelFirstd(keys=["image"]),  # 确保图像有通道维度
            Resized(keys=["image", "label"], spatial_size=config.RESIZE_SHAPE),
            NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"], device=config.DEVICE, dtype=torch.float32)
        ])

        return final_transforms(data_dict)


def get_data_loaders():
    """
    创建训练和验证数据加载器
    """
    # 1. 加载标签文件
    with open(config.LABEL_FILE, 'r') as f:
        all_data_info = json.load(f)

    # 2. 划分训练/验证集
    num_total = len(all_data_info)
    num_val = int(num_total * config.VAL_SPLIT)
    num_train = num_total - num_val

    indices = torch.randperm(num_total).tolist()
    train_info = [all_data_info[i] for i in indices[:num_train]]
    val_info = [all_data_info[i] for i in indices[num_train:]]

    # 3. 定义 (空的) 变换
    # 因为我们在 Dataset __getitem__ 中处理了所有逻辑
    # (这是一个简化的设计，通常你会把所有变换放在这里)
    # 让我们遵循上面的逻辑

    train_ds = CTScanDataset(train_info, config.DATA_DIR, transforms=None)
    val_ds = CTScanDataset(val_info, config.DATA_DIR, transforms=None)

    # 4. 创建 DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Windows/Jupyter 设为 0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


# (可选) 测试数据加载器
if __name__ == "__main__":
    print("测试数据加载器...")
    # 确保您有 "data/images" 和 "data/labels.json"
    try:
        train_loader, _ = get_data_loaders()
        batch = first(train_loader)

        print(f"设备: {batch['image'].device}")
        print(f"图像批次形状: {batch['image'].shape}")
        print(f"标签批次形状: {batch['label'].shape}")

        # 检查标签是否正确生成
        # (B, NumClasses, D, H, W)
        print(f"标签总和 (检查是否全为0): {batch['label'].sum()}")
        assert batch['label'].sum() > 0, "掩码未正确生成，检查 labels.json 和 utils.py"

        print("数据加载器测试成功!")

    except FileNotFoundError:
        print("\n[错误] 未找到 'data/labels.json' 或 'data/images' 目录。")
        print("请按照说明创建 data 目录和 labels.json 文件。")
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        print("请检查您的 `labels.json` 格式和 `utils.py` 中的 `create_mask_from_bboxes` 逻辑。")