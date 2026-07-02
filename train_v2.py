import pickle
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from bpr_model import BPRModel
from bpr_dataset import BPRDataset

# ===================== 配置 =====================
LATENT_DIM = 128
BATCH_SIZE = 2048      # 如果 CPU 训练很慢，可以进一步降低到 1024 或 512
EPOCHS = 30
LR = 1e-3
# ===============================================

if __name__ == '__main__':
    print("Loading training data...", flush=True)
    try:
        with open("train_positive.pkl", "rb") as f:
            train_positive = pickle.load(f)
        with open("movie2idx.pkl", "rb") as f:
            movie2idx = pickle.load(f)
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e}. 请先运行 BPR1.py 生成数据。", flush=True)
        exit(1)

    if not train_positive:
        print("错误：train_positive 为空，请检查 BPR1.py 是否生成了有效的训练数据。", flush=True)
        exit(1)

    n_users = max(train_positive.keys()) + 1
    n_items = len(movie2idx)
    print(f"用户数: {n_users}, 物品数: {n_items}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)

    print("创建数据集...", flush=True)
    dataset = BPRDataset(train_positive, n_items)

    print("创建 DataLoader...", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # Windows 安全，避免死锁
        pin_memory=False        # CPU 训练时无用，关闭
    )

    print("初始化模型...", flush=True)
    model = BPRModel(n_users, n_items, LATENT_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    def bpr_loss(pos_score, neg_score):
        return -torch.mean(
            torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
        )

    print("开始训练...", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (users, pos, neg) in enumerate(loader):
            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_score, neg_score = model(users, pos, neg)
            loss = bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 每 10 个 batch 打印一次，让你看到进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1:2d} | Batch {batch_idx:4d} | loss = {loss.item():.6f}", flush=True)

        avg_loss = total_loss / num_batches
        print(f"==> Epoch {epoch+1} 完成，平均 Loss = {avg_loss:.6f}", flush=True)

        # 每个 epoch 保存一次
        torch.save(model.state_dict(), f"bpr_epoch_{epoch+1}.pt")
        print(f"    已保存 bpr_epoch_{epoch+1}.pt", flush=True)

    # 最终保存最佳模型
    torch.save(model.state_dict(), "best_model.pt")
    print("训练完成！最佳模型已保存为 best_model.pt", flush=True)
