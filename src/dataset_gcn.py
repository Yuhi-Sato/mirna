import os
import torch
from dataset_predict import (
    features,
    functional_similarity,
)
import numpy as np
from n_grams import f
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index
import random
from path import data_dir


class Disease:
    def __init__(self, path, features):
        disease_file_list = os.listdir(path)
        disease_file_list.sort()

        self.disease_class = []
        for index, disease_file_name in enumerate(disease_file_list):
            self.disease_class.append([])
            with open(path + disease_file_name, "r") as f:
                for line in f:
                    line = line.rstrip().split("\t")
                    mirna_name = line[0]
                    if (
                        mirna_name in features
                        and not mirna_name in self.disease_class[index]
                    ):
                        self.disease_class[index].append(mirna_name)

    def get_specific_disease_y(self, features, disease_type: int):
        y = [0] * len(features)
        for index, mir in enumerate(features):
            if mir in self.disease_class[disease_type]:
                y[index] = 1
        return torch.tensor(y)

    def disease_type_to_name(self, disease_type: int):
        disease_file_list = os.listdir(data_dir + "/disease/")
        disease_file_list.sort()
        return disease_file_list[disease_type][:-4]

    # NOTE: rateはラベル1に対してのラベル0の割合
    def get_specific_disease_train_mask(
        self, features, disease_type: int, rate, seed: int
    ):
        y = self.get_specific_disease_y(features, disease_type)

        train_mask = torch.tensor([True] * len(y), dtype=torch.bool)
        train_mask[torch.where(y == 0)] = False

        current_true = train_mask.sum()
        reverse_count = int(current_true * rate)

        if current_true + reverse_count > len(y):
            reverse_count = len(y) - current_true

        target = current_true + reverse_count
        cnt = 0
        random.seed(seed)
        list = [i for i in range(len(y))]
        random.shuffle(list)
        for i in list:
            if train_mask[i] == False:
                cnt += 1
                train_mask[i] = True
            if cnt == target:
                break

        return train_mask


def get_data(disease_type: int, seed: int) -> Data:
    A = torch.from_numpy(functional_similarity.similarity).to_sparse()
    tuple = to_edge_index(A)
    edge_index, edge_attr = tuple[0], tuple[1]

    disease = Disease(path=data_dir + "/disease/", features=features)

    X = torch.tensor(f, dtype=torch.float)
    y = disease.get_specific_disease_y(features, disease_type)
    train_mask = disease.get_specific_disease_train_mask(
        features, disease_type, rate=1.0, seed=seed
    )

    data = Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_attr.to(torch.float),
        y=y,
        train_mask=train_mask,
    )

    return data
