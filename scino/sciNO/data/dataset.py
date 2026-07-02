# data/dataset.py
"""数据集工具"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDAGDataset(Dataset):
    """合成DAG数据集"""
    def __init__(self, n_nodes, n_samples, p=0.08, seed=None):
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        if seed is not None:
            np.random.seed(seed)
        self.X, self.W = self._generate(p)

    def _generate(self, p):
        # 生成稀疏上三角邻接矩阵
        A = (np.random.rand(self.n_nodes, self.n_nodes) < p).astype(float)
        A = np.triu(A, k=1)
        W = A * (np.random.randn(*A.shape) * 0.8)

        # 根据DAG生成数据
        X = np.zeros((self.n_samples, self.n_nodes), dtype=float)
        for i in range(self.n_nodes):
            parents = np.where(W[:, i] != 0)[0]
            if len(parents) == 0:
                X[:, i] = np.random.randn(self.n_samples) * 0.5
            else:
                X[:, i] = np.tanh(X[:, parents].dot(W[parents, i])) + \
                          0.1 * np.random.randn(self.n_samples)

        return X.astype('float32'), W

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx]


class TensorDataset(Dataset):
    """简单张量数据集包装器"""
    def __init__(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
