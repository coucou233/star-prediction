import pickle
import torch
import numpy as np
from tqdm import tqdm
import csv
import time

# ===================== 配置 =====================
K_VALUES = [5, 10, 20, 50]          # 评估的 K 值
BATCH_SIZE = 64                    # GPU 显存允许的批大小（可调）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===================== 加载数据 =====================
print("加载数据...")
with open("train_positive.pkl", "rb") as f:
    train_positive = pickle.load(f)
with open("test_positive.pkl", "rb") as f:
    test_positive = pickle.load(f)
with open("user2idx.pkl", "rb") as f:
    user2idx = pickle.load(f)
with open("movie2idx.pkl", "rb") as f:
    movie2idx = pickle.load(f)

user_list = list(test_positive.keys())
print(f"总用户数: {len(user_list)}")

n_users = len(user2idx)
n_items = len(movie2idx)

# 反向映射（用于保存推荐列表，可选，如果只需指标可不保存）
SAVE_RECS = False   # 是否保存详细推荐列表（全量保存文件会很大）

# ===================== 加载模型 =====================
from bpr_model import BPRModel
model = BPRModel(n_users, n_items)
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 预计算所有物品的嵌入和偏置
with torch.no_grad():
    all_item_emb = model.item_emb.weight.to(DEVICE)          # (n_items, dim)
    all_item_bias = model.item_bias.weight.squeeze(-1).to(DEVICE)  # (n_items,)

# ===================== 统计变量 =====================
num_users = len(user_list)
hits = {k: 0 for k in K_VALUES}
ndcgs = {k: 0.0 for k in K_VALUES}
total_rank = 0
total_mrr = 0.0
max_k = max(K_VALUES)

# 用于记录排名分布（可选，抽样记录）
sample_ranks = []

start_time = time.time()

# ===================== 分批处理 =====================
for start_idx in tqdm(range(0, num_users, BATCH_SIZE), desc="Evaluating full users"):
    batch_users = user_list[start_idx:start_idx + BATCH_SIZE]
    batch_size = len(batch_users)
    batch_user_idx = torch.tensor(batch_users, dtype=torch.long, device=DEVICE)

    # 获取用户嵌入和偏置
    user_emb = model.user_emb(batch_user_idx)                # (batch, dim)
    user_bias = model.user_bias(batch_user_idx).squeeze(-1)  # (batch,)

    # 计算所有物品的分数: (batch, n_items)
    scores = torch.matmul(user_emb, all_item_emb.T) + user_bias.unsqueeze(1) + all_item_bias.unsqueeze(0)

    # 获取这批用户的真实物品索引
    true_items = torch.tensor([test_positive[u] for u in batch_users], device=DEVICE)  # (batch,)

    # 提取真实物品的分数
    true_scores = scores.gather(1, true_items.unsqueeze(1)).squeeze(1)  # (batch,)

    # 屏蔽训练物品（但绝不屏蔽真实物品）
    # 对于每个用户，将其训练物品集合中的（非真实物品）分数设为 -inf
    for i, u in enumerate(batch_users):
        if u in train_positive:
            # 获取训练物品列表，并确保不包含真实物品
            to_mask = [it for it in train_positive[u] if it != true_items[i].item()]
            if to_mask:
                scores[i, to_mask] = -float('inf')

    # 重新计算真实物品的分数（可能被屏蔽，但我们已经排除了真实物品，所以不需要重新获取）
    # 但为了安全，再次提取（因为屏蔽操作不影响真实物品位置）
    true_scores = scores.gather(1, true_items.unsqueeze(1)).squeeze(1)

    # 计算排名：分数高于真实物品分数的物品数量 + 1
    # 注意：屏蔽后其他物品分数可能为 -inf，不影响比较
    higher_mask = scores > true_scores.unsqueeze(1)   # (batch, n_items)
    ranks = higher_mask.sum(dim=1) + 1                # (batch,)

    # 更新总排名和 MRR
    total_rank += ranks.sum().item()
    total_mrr += (1.0 / ranks.float()).sum().item()

    # 记录部分排名用于百分位分析（最多记录10000个）
    if len(sample_ranks) < 10000:
        sample_ranks.extend(ranks.cpu().numpy().tolist())

    # 计算 Top-K 指标：需要对每个用户获取前 max_k 个物品
    # 使用 torch.topk 获取前 max_k 个索引（分数已屏蔽训练物品，且真实物品不会被屏蔽）
    topk_scores, topk_indices = torch.topk(scores, max_k, dim=1)  # (batch, max_k)

    for i in range(batch_size):
        true_item = true_items[i].item()
        topk = topk_indices[i].cpu().numpy()
        rank_i = ranks[i].item()

        for k in K_VALUES:
            if true_item in topk[:k]:
                hits[k] += 1
                pos = np.where(topk[:k] == true_item)[0][0]   # 0-indexed
                ndcgs[k] += 1.0 / np.log2(pos + 2)

    # 可选：保存推荐列表（文件会很大，全量约 48万行 x 50 个电影 ID）
    if SAVE_RECS:
        # 省略保存代码，需要时可开启，但建议只保存前1000用户示例
        pass

end_time = time.time()
elapsed = end_time - start_time
print(f"\n评估完成，耗时 {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")

# ===================== 输出结果 =====================
avg_rank = total_rank / num_users
mrr = total_mrr / num_users

print("\n========== 全量用户 Top-K 评估结果 ==========")
print(f"用户数: {num_users}")
print(f"平均排名 (Average Rank): {avg_rank:.2f}")
print(f"MRR: {mrr:.4f}")

if sample_ranks:
    percentiles = np.percentile(sample_ranks, [10, 50, 90])
    print(f"排名百分位数 (抽样 {len(sample_ranks)} 用户): 10%={percentiles[0]:.1f}, 50%={percentiles[1]:.1f}, 90%={percentiles[2]:.1f}")

print("\nK\tHit Rate\tNDCG")
for k in sorted(K_VALUES):
    hit_rate = hits[k] / num_users
    ndcg = ndcgs[k] / num_users
    print(f"{k}\t{hit_rate:.4f}\t\t{ndcg:.4f}")
