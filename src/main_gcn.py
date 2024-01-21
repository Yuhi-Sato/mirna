from gcn import Net
import torch
import torch.nn.functional as F
from dataset_gcn import get_data, features
import numpy as np
from path import root_dir
import hashlib


def custom_hash(x, y):
    data = f"{x}-{y}".encode("utf-8")
    hash_object = hashlib.sha256(data)
    hashed_value = hash_object.hexdigest()
    return hashed_value


def main(disease_type: int, seed: int) -> torch.Tensor:
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

    data = get_data(disease_type, seed=seed).to(device)

    model = Net(num_node_features=data.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(
            out[data.train_mask], data.y[data.train_mask].float()
        )
        loss.backward()
        optimizer.step()

    model.eval()

    y = model(data)

    return y


def bagging(disease_type, n):
    # 多数決でラベルを決定する
    z = np.zeros((n, len(features)))
    for i in range(n):
        y = main(disease_type=disease_type, seed=custom_hash(disease_type, i))
        print(y)
        for j in range(len(y)):
            z[i][j] = 1 if y[j] >= 0.5 else 0

    # 0, 1の多数決
    z = np.sum(z, axis=0)
    z = np.where(z > n / 2, 1, 0)

    return z


if __name__ == "__main__":
    for i in range(28):
        file_name = root_dir + "/results_gcn_hash/" + str(i) + ".csv"
        pred = bagging(i, 10)
        with open(file_name, "w") as f:
            for _, p in enumerate(pred):
                f.write(str(p.item()) + "\n")
