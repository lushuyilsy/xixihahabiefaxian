import torch.nn as nn
from monai.networks.nets import UNet
import config


def get_model():
    """
    实例化 3D U-Net 模型
    """
    model = UNet(
        spatial_dims=3,  # 3D
        in_channels=1,  # 输入通道 (CT 图像)
        out_channels=config.NUM_CLASSES,  # 输出通道 (每个 ROI 类别一个)
        channels=(16, 32, 64, 128, 256),  # 网络深度和宽度
        strides=(2, 2, 2, 2),
        num_res_units=2
    )

    # 我们的模型输出 (B, NumClasses, D, H, W)
    # 我们将使用 BCEWithLogitsLoss 或 DiceLoss，
    # 它们内部处理了 Sigmoid/Softmax

    return model.to(config.DEVICE)


if __name__ == "__main__":
    model = get_model()
    print("模型架构:")
    print(model)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")