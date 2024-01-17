from gcn2 import Net
import torch
import torch.nn.functional as F
from dataset_gcn import get_data, features
import numpy as np
from path import root_dir


def main(disease_type: int, seed: int) -> torch.Tensor:
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")

    data = get_data(disease_type, seed=seed).to(device)

    model = Net(num_node_features=data.x.shape[1], num_layers=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # NOTE: 訓練
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        print("epoch:", epoch)
        print(out)
        loss = F.binary_cross_entropy(
            out[data.train_mask], data.y[data.train_mask].float()
        )
        loss.backward()
        optimizer.step()

    # NOTE: テスト
    model.eval()

    y = model(data)

    return y


def bagging(disease_type, n):
    # 多数決でラベルを決定する
    z = np.zeros((n, len(features)))
    for i in range(n):
        y = main(disease_type=disease_type, seed=i)
        print("i-th:", y)
        for j in range(len(y)):
            z[i][j] = 1 if y[j] >= 0.5 else 0

    # 0, 1の多数決
    z = np.sum(z, axis=0)
    z = np.where(z > n / 2, 1, 0)

    return z


if __name__ == "__main__":
    for i in range(28):
        file_name = root_dir + "/results_gcn2/" + str(i) + ".csv"
        pred = bagging(i, 10)
        with open(file_name, "w") as f:
            for _, p in enumerate(pred):
                f.write(str(p.item()) + "\n")
