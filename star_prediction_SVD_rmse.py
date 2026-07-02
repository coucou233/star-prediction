# ==========================================
# Netflix Prize Bias-SVD 推荐系统（终极加速稳定版）
# ==========================================

# 优化内容：
# 1. 修复所有语法错误
# 2. float32 降低内存
# 3. NumPy数组加速
# 4. 去掉copy()
# 5. 只shuffle一次
# 6. 更低latent factor
# 7. 更少epoch
# 8. 自动检测路径
# 9. 防止空数据崩溃
# 10. 输出运行时间
# ==========================================

import numpy as np
import random
import time
import os

# ==========================================
# 参数设置
# ==========================================

TRAIN_FOLDER = r"training_set"

QUALIFY_FILE = "qualifying.txt"

OUTPUT_FILE = "submission.txt"

LATENT_FACTORS = 20

LEARNING_RATE = 0.005

REGULARIZATION = 0.02

EPOCHS = 8

SEED = 42

np.random.seed(SEED)

random.seed(SEED)

# ==========================================
# 数据读取
# ==========================================

def load_training_data(folder_path):

    print("=" * 50)
    print("开始读取训练数据...")
    print("=" * 50)

    print("当前工作目录:")
    print(os.getcwd())

    print("训练集路径:")
    print(folder_path)

    # 检查路径

    if not os.path.exists(folder_path):

        print(f"错误：目录不存在 -> {folder_path}")

        return [], set(), set()

    # 读取txt文件

    files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".txt")
    ]

    print(f"发现 {len(files)} 个 txt 文件")

    if len(files) == 0:

        print("错误：没有找到 txt 文件")

        return [], set(), set()

    data = []

    user_set = set()

    movie_set = set()

    total_ratings = 0

    # 遍历文件

    for file_idx, file_name in enumerate(files):

        file_path = os.path.join(
            folder_path,
            file_name
        )

        current_movie = None

        with open(
            file_path,
            'r',
            encoding='utf-8'
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                # MovieID:
                if line.endswith(':'):

                    current_movie = int(
                        line[:-1]
                    )

                    movie_set.add(current_movie)

                else:

                    parts = line.split(',')

                    user_id = int(parts[0])

                    rating = float(parts[1])

                    user_set.add(user_id)

                    data.append(
                        (
                            user_id,
                            current_movie,
                            rating
                        )
                    )

                    total_ratings += 1

                    if total_ratings % 1000000 == 0:

                        print(
                            f"已读取 {total_ratings:,} 条评分"
                        )

        print(
            f"完成文件 "
            f"{file_idx+1}/{len(files)} : "
            f"{file_name}"
        )

    print("=" * 50)
    print("训练数据读取完成")
    print("=" * 50)

    print(f"总评分数: {len(data):,}")

    print(f"用户数: {len(user_set):,}")

    print(f"电影数: {len(movie_set):,}")

    return data, user_set, movie_set

# ==========================================
# ID映射
# ==========================================

def build_id_mapping(user_set, movie_set):

    print("=" * 50)
    print("构建ID映射...")
    print("=" * 50)

    user2idx = {}

    idx2user = {}

    for idx, user_id in enumerate(user_set):

        user2idx[user_id] = idx

        idx2user[idx] = user_id

    movie2idx = {}

    idx2movie = {}

    for idx, movie_id in enumerate(movie_set):

        movie2idx[movie_id] = idx

        idx2movie[idx] = movie_id

    print("ID映射完成")

    return (
        user2idx,
        idx2user,
        movie2idx,
        idx2movie
    )

# ==========================================
# 编码训练数据
# ==========================================

def encode_training_data(
        data,
        user2idx,
        movie2idx):

    print("=" * 50)
    print("编码训练数据...")
    print("=" * 50)

    n = len(data)

    users = np.zeros(
        n,
        dtype=np.int32
    )

    items = np.zeros(
        n,
        dtype=np.int32
    )

    ratings = np.zeros(
        n,
        dtype=np.float32
    )

    for idx, (user_id, movie_id, rating) in enumerate(data):

        users[idx] = user2idx[user_id]

        items[idx] = movie2idx[movie_id]

        ratings[idx] = rating

        if idx % 5000000 == 0 and idx > 0:

            print(
                f"已编码 {idx:,} 条"
            )

    print("训练数据编码完成")

    return users, items, ratings

# ==========================================
# Bias-SVD
# ==========================================

class BiasSVD:

    def __init__(self,
                 n_users,
                 n_items,
                 k=20,
                 lr=0.005,
                 reg=0.02,
                 epochs=8):

        self.n_users = n_users

        self.n_items = n_items

        self.k = k

        self.lr = lr

        self.reg = reg

        self.epochs = epochs

        print("=" * 50)
        print("初始化模型...")
        print("=" * 50)

        # 用户隐向量

        self.P = np.random.normal(
            0,
            0.1,
            (n_users, k)
        ).astype(np.float32)

        # 电影隐向量

        self.Q = np.random.normal(
            0,
            0.1,
            (n_items, k)
        ).astype(np.float32)

        # bias

        self.bu = np.zeros(
            n_users,
            dtype=np.float32
        )

        self.bi = np.zeros(
            n_items,
            dtype=np.float32
        )

        self.global_mean = 0.0

        print("模型初始化完成")

    # ======================================
    # 单个预测
    # ======================================

    def predict_single(self, u, i):

        pred = (
            self.global_mean
            + self.bu[u]
            + self.bi[i]
            + np.dot(
                self.P[u],
                self.Q[i]
            )
        )

        if pred < 1:
            pred = 1

        elif pred > 5:
            pred = 5

        return pred

    # ======================================
    # 训练
    # ======================================

    def train(
            self,
            users,
            items,
            ratings):

        print("=" * 50)
        print("开始训练 Bias-SVD...")
        print("=" * 50)

        n = len(ratings)

        if n == 0:

            print("错误：训练数据为空")

            return

        self.global_mean = np.mean(ratings)

        print(
            f"全局平均分: "
            f"{self.global_mean:.4f}"
        )

        print(f"训练样本数: {n:,}")

        # 只shuffle一次

        print("正在shuffle数据...")

        indices = np.arange(n)

        np.random.shuffle(indices)

        users = users[indices]

        items = items[indices]

        ratings = ratings[indices]

        print("shuffle完成")

        # 开始epoch

        for epoch in range(self.epochs):

            print("=" * 50)

            print(
                f"开始 Epoch "
                f"{epoch+1}/{self.epochs}"
            )

            print("=" * 50)

            start_time = time.time()

            total_loss = 0.0

            for idx in range(n):

                u = users[idx]

                i = items[idx]

                r = ratings[idx]

                # 预测

                pred = (
                    self.global_mean
                    + self.bu[u]
                    + self.bi[i]
                    + np.dot(
                        self.P[u],
                        self.Q[i]
                    )
                )

                # clip

                if pred < 1:
                    pred = 1

                elif pred > 5:
                    pred = 5

                err = r - pred

                total_loss += err * err

                # 更新 bias

                self.bu[u] += self.lr * (
                    err
                    - self.reg * self.bu[u]
                )

                self.bi[i] += self.lr * (
                    err
                    - self.reg * self.bi[i]
                )

                # 不copy()

                Pu = self.P[u]

                Qi = self.Q[i]

                # 更新隐向量

                self.P[u] += self.lr * (
                    err * Qi
                    - self.reg * Pu
                )

                self.Q[i] += self.lr * (
                    err * Pu
                    - self.reg * Qi
                )

                # 进度输出

                if idx % 5000000 == 0 and idx > 0:

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    print(
                        f"Epoch {epoch+1} | "
                        f"已训练 {idx:,}/{n:,} | "
                        f"耗时 {elapsed:.2f} 秒"
                    )

            rmse = np.sqrt(
                total_loss / n
            )

            end_time = time.time()

            print("=" * 50)

            print(
                f"Epoch {epoch+1}/{self.epochs} 完成"
            )

            print(
                f"RMSE = {rmse:.4f}"
            )

            print(
                f"Epoch耗时 = "
                f"{end_time-start_time:.2f} 秒"
            )

            print("=" * 50)

# ==========================================
# 读取qualifying.txt
# ==========================================

def load_qualifying_file(filename):

    print("=" * 50)
    print("读取 qualifying 文件...")
    print("=" * 50)

    queries = []

    current_movie = None

    with open(
        filename,
        'r',
        encoding='utf-8'
    ) as f:

        for line in f:

            line = line.strip()

            if line.endswith(':'):

                current_movie = int(
                    line[:-1]
                )

            else:

                user_id = int(
                    line.split(',')[0]
                )

                queries.append(
                    (
                        user_id,
                        current_movie
                    )
                )

    print(
        f"待预测数量: {len(queries):,}"
    )

    return queries

# ==========================================
# 生成submission
# ==========================================

def generate_submission(
        model,
        queries,
        user2idx,
        movie2idx,
        output_file):

    print("=" * 50)
    print("生成 submission...")
    print("=" * 50)

    with open(
        output_file,
        'w',
        encoding='utf-8'
    ) as f:

        last_movie = None

        for idx, (user_id, movie_id) in enumerate(queries):

            # MovieID:

            if movie_id != last_movie:

                f.write(f"{movie_id}:\n")

                last_movie = movie_id

            # 冷启动

            if (
                user_id not in user2idx
                or movie_id not in movie2idx
            ):

                pred = model.global_mean

            else:

                u = user2idx[user_id]

                i = movie2idx[movie_id]

                pred = model.predict_single(u, i)

            f.write(f"{pred:.4f}\n")

            if idx % 1000000 == 0 and idx > 0:

                print(
                    f"已预测 {idx:,} 条"
                )

    print(
        f"submission 已保存: "
        f"{output_file}"
    )

# ==========================================
# 主函数
# ==========================================

def main():

    total_start = time.time()

    # 1. 读取数据

    raw_data, user_set, movie_set = load_training_data(
        TRAIN_FOLDER
    )

    # 防呆

    if len(raw_data) == 0:

        print("错误：没有读取到训练数据")

        return

    # 2. ID映射

    (
        user2idx,
        idx2user,
        movie2idx,
        idx2movie
    ) = build_id_mapping(
        user_set,
        movie_set
    )

    # 3. 编码

    users, items, ratings = encode_training_data(
        raw_data,
        user2idx,
        movie2idx
    )

    # 释放内存

    del raw_data

    # 4. 初始化模型

    model = BiasSVD(
        n_users=len(user2idx),
        n_items=len(movie2idx),
        k=LATENT_FACTORS,
        lr=LEARNING_RATE,
        reg=REGULARIZATION,
        epochs=EPOCHS
    )

    # 5. 训练

    model.train(
        users,
        items,
        ratings
    )

    # 6. 读取qualifying

    queries = load_qualifying_file(
        QUALIFY_FILE
    )

    # 7. 生成submission

    generate_submission(
        model,
        queries,
        user2idx,
        movie2idx,
        OUTPUT_FILE
    )

    total_end = time.time()

    print("=" * 50)
    print("全部完成")
    print("=" * 50)

    print(
        f"总耗时: "
        f"{total_end-total_start:.2f} 秒"
    )

# ==========================================
# 程序入口
# ==========================================

if __name__ == '__main__':

    main()

