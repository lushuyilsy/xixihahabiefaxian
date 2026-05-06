import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from model import get_model
from data_loader import get_data_loaders
from monai.losses import DiceLoss


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(config.DEVICE)  # (B, 1, D, H, W)
        labels = batch["label"].to(config.DEVICE)  # (B, NumClasses, D, H, W)

        # 1. Forward
        optimizer.zero_grad()
        outputs = model(images)  # (B, NumClasses, D, H, W)

        # 2. Loss
        # DiceLoss 期望 (B, C, ...) 和 (B, C, ...)
        # 我们使用 Sigmoid 来进行多标签分割
        loss = loss_fn(torch.sigmoid(outputs), labels)

        # 3. Backward
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return epoch_loss / len(loader)


def validate_epoch(model, loader, loss_fn):
    model.eval()
    epoch_loss = 0.0

    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)

            outputs = model(images)
            loss = loss_fn(torch.sigmoid(outputs), labels)

            epoch_loss += loss.item()
            pbar.set_postfix({"val_loss": loss.item()})

    return epoch_loss / len(loader)


def main():
    print(f"开始训练，使用设备: {config.DEVICE}")

    # 1. 获取模型
    model = get_model()

    # 2. 获取数据加载器
    try:
        train_loader, val_loader = get_data_loaders()
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请确保 'data/images' 目录和 'data/labels.json' 文件已准备就绪。")
        return

    # 3. 定义损失函数和优化器
    # DiceLoss 适用于分割任务
    loss_fn = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float('inf')

    # 4. 训练循环
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = validate_epoch(model, val_loader, loss_fn)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 5. 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"模型已保存到: {config.MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})")

    print("训练完成。")


if __name__ == "__main__":
    main()