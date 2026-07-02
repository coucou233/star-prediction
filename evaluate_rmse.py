import pickle
import torch
from bpr_model import BPRModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

if __name__ == '__main__':
    # 加载映射和模型
    with open("movie2idx.pkl", "rb") as f:
        movie2idx = pickle.load(f)
    with open("user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)
    
    # 加载 RMSE 样本（原始评分）
    with open("rmse_samples.pkl", "rb") as f:
        rmse_samples = pickle.load(f)   # list of (user_id, movie_id, rating)
    
    n_users = len(user2idx)
    n_items = len(movie2idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BPRModel(n_users, n_items)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # 准备批量预测
    user_ids = []
    item_ids = []
    true_ratings = []
    
    for uid, mid, rating in rmse_samples:
        if uid in user2idx and mid in movie2idx:
            user_ids.append(user2idx[uid])
            item_ids.append(movie2idx[mid])
            true_ratings.append(rating)
    
    # 转为 tensor
    user_tensor = torch.tensor(user_ids, device=device)
    item_tensor = torch.tensor(item_ids, device=device)
    
    # 分批预测（避免内存爆炸）
    batch_size = 8192
    preds = []
    with torch.no_grad():
        for i in range(0, len(user_tensor), batch_size):
            batch_u = user_tensor[i:i+batch_size]
            batch_i = item_tensor[i:i+batch_size]
            pred = model.score(batch_u, batch_i)   # shape (batch,)
            preds.append(pred.cpu().numpy())
    
    pred_ratings = np.concatenate(preds)
    true_ratings = np.array(true_ratings)
    
    # 计算 RMSE
    mse = np.mean((pred_ratings - true_ratings) ** 2)
    rmse = np.sqrt(mse)
    print(f"RMSE on {len(true_ratings)} samples: {rmse:.4f}")
