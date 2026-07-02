import pickle
import random
import torch
from bpr_model import BPRModel


import os
import requests
import torch
import pickle
from tqdm import tqdm

# ============================
# 1. 云端文件直链（
# ============================
MODEL_URL = "https://star-prediction-model.oss-cn-shanghai.aliyuncs.com/best_model.pt"
TRAIN_DATA_URL = "https://star-prediction-model.oss-cn-shanghai.aliyuncs.com/train_positive.pkl"

MODEL_PATH = "best_model.pt"
TRAIN_DATA_PATH = "train_positive.pkl"

# ============================
# 2. 通用下载函数（带进度条）
# ============================
def download_file(url, filepath):
    """从指定URL下载文件到本地，并显示进度条"""
    if os.path.exists(filepath):
        print(f"✅ 本地文件已存在: {filepath}")
        return
    
    print(f"📥 正在下载: {filepath} (约 {int(requests.head(url).headers.get('content-length', 0)) / 1024 / 1024} MB)")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"✅ 下载完成: {filepath}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        raise

# ============================
# 3. 下载两个大文件（如果本地不存在）
# ============================
download_file(MODEL_URL, MODEL_PATH)
download_file(TRAIN_DATA_URL, TRAIN_DATA_PATH)


# 配置
NEGATIVE_SAMPLES = 50          # 先减少负样本数，加快评估（可改回100）
PRINT_EVERY = 1000             # 每处理1000个用户打印一次进度

if __name__ == '__main__':
    print("Loading data...", flush=True)
    
    with open("train_positive.pkl", "rb") as f:
        train_positive = pickle.load(f)
    with open("test_positive.pkl", "rb") as f:
        test_positive = pickle.load(f)
    with open("movie2idx.pkl", "rb") as f:
        movie2idx = pickle.load(f)
    
    n_users = max(train_positive.keys()) + 1
    n_items = len(movie2idx)
    print(f"Users: {n_users}, Items: {n_items}", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    model = BPRModel(n_users, n_items)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    
    print("Start evaluating AUC...", flush=True)
    
    correct = 0
    total = 0
    user_count = 0
    
    with torch.no_grad():
        for u, pos_item in test_positive.items():
            pos_tensor = torch.tensor([u], device=device)
            pos_item_tensor = torch.tensor([pos_item], device=device)
            pos_score = model.score(pos_tensor, pos_item_tensor).item()
            
            # 负采样
            neg_items = set()
            while len(neg_items) < NEGATIVE_SAMPLES:
                neg = random.randint(0, n_items - 1)
                if neg not in train_positive[u] and neg != pos_item:  # 确保不是正样本
                    neg_items.add(neg)
            
            for neg in neg_items:
                neg_score = model.score(
                    torch.tensor([u], device=device),
                    torch.tensor([neg], device=device)
                ).item()
                if pos_score > neg_score:
                    correct += 1
                total += 1
            
            user_count += 1
            if user_count % PRINT_EVERY == 0:
                print(f"Processed {user_count} users, current AUC = {correct/total:.4f}", flush=True)
    
    auc = correct / total
    print(f"\nFinal AUC = {auc:.6f} ({correct}/{total})", flush=True)
