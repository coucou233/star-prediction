
# =====================================================
# 01_preprocess.py
# Netflix Prize -> BPR Dataset
# =====================================================

import os
import pickle
import random
from collections import defaultdict
from datetime import datetime

# =====================================================
# 参数
# =====================================================

DATA_DIR = r"D:\nf_prize_dataset.tar\download\download\training_set"

POSITIVE_THRESHOLD = 4

RMSE_SAMPLE_SIZE = 1_000_000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# =====================================================
# 数据结构
# =====================================================

user_positive_raw = defaultdict(list)

user_set = set()
movie_set = set()

rmse_samples = []

total_ratings = 0

# =====================================================
# 遍历文件
# =====================================================

files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".txt")
])

print("=" * 60)
print("Start Reading Netflix Dataset")
print(f"Movie files: {len(files)}")
print("=" * 60)

for file_idx, file_name in enumerate(files):

    path = os.path.join(DATA_DIR, file_name)

    current_movie = None

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # MovieID:
            if line.endswith(":"):

                current_movie = int(
                    line[:-1]
                )

                movie_set.add(current_movie)

                continue

            parts = line.split(",")

            user_id = int(parts[0])

            rating = int(parts[1])

            date_str = parts[2]

            user_set.add(user_id)

            total_ratings += 1

            # ---------------------------------
            # Positive feedback
            # ---------------------------------

            if rating >= POSITIVE_THRESHOLD:

                user_positive_raw[user_id].append(
                    (
                        current_movie,
                        date_str
                    )
                )

            # ---------------------------------
            # RMSE Sample Reservoir
            # ---------------------------------

            sample_item = (
                user_id,
                current_movie,
                rating
            )

            if len(rmse_samples) < RMSE_SAMPLE_SIZE:

                rmse_samples.append(sample_item)

            else:

                r = random.randint(
                    0,
                    total_ratings - 1
                )

                if r < RMSE_SAMPLE_SIZE:

                    rmse_samples[r] = sample_item

    if (file_idx + 1) % 100 == 0:

        print(
            f"Processed "
            f"{file_idx+1}/{len(files)} files"
        )

print()
print("=" * 60)
print("Finished Reading")
print("=" * 60)

print(f"Ratings : {total_ratings:,}")
print(f"Users   : {len(user_set):,}")
print(f"Movies  : {len(movie_set):,}")

# =====================================================
# 建立映射
# =====================================================

print()
print("Building ID Mapping...")

user_list = sorted(list(user_set))
movie_list = sorted(list(movie_set))

user2idx = {
    u: i
    for i, u in enumerate(user_list)
}

movie2idx = {
    m: i
    for i, m in enumerate(movie_list)
}

print("User Mapping Done")
print("Movie Mapping Done")

# =====================================================
# Leave-One-Out Split
# =====================================================

print()
print("Creating Train/Test Positive Set...")

train_positive = defaultdict(set)

test_positive = dict()

valid_users = 0

for user_id, movie_records in user_positive_raw.items():

    if len(movie_records) < 2:
        continue

    movie_records.sort(
        key=lambda x:
        datetime.strptime(
            x[1],
            "%Y-%m-%d"
        )
    )

    test_movie = movie_records[-1][0]

    train_movies = [
        x[0]
        for x in movie_records[:-1]
    ]

    u = user2idx[user_id]

    test_positive[u] = movie2idx[test_movie]

    for movie_id in train_movies:

        train_positive[u].add(
            movie2idx[movie_id]
        )

    valid_users += 1

print(
    f"Valid Users: {valid_users:,}"
)

# =====================================================
# 保存
# =====================================================

print()
print("Saving Files...")

with open(
    "user2idx.pkl",
    "wb"
) as f:

    pickle.dump(
        user2idx,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

with open(
    "movie2idx.pkl",
    "wb"
) as f:

    pickle.dump(
        movie2idx,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

with open(
    "train_positive.pkl",
    "wb"
) as f:

    pickle.dump(
        dict(train_positive),
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

with open(
    "test_positive.pkl",
    "wb"
) as f:

    pickle.dump(
        test_positive,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

with open(
    "rmse_samples.pkl",
    "wb"
) as f:

    pickle.dump(
        rmse_samples,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print()
print("=" * 60)
print("Preprocess Completed")
print("=" * 60)

print("Generated Files:")
print("user2idx.pkl")
print("movie2idx.pkl")
print("train_positive.pkl")
print("test_positive.pkl")
print("rmse_samples.pkl")

