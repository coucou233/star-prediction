
import random
import pickle
from torch.utils.data import Dataset

class BPRDataset(Dataset):
    def __init__(self, train_positive, n_items):
        self.train_positive = train_positive
        self.n_items = n_items
        self.users = list(train_positive.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]

        pos = random.choice(tuple(self.train_positive[u]))

        while True:
            neg = random.randint(0, self.n_items - 1)
            if neg not in self.train_positive[u]:
                break

        return u, pos, neg
