import numpy as np
from path import data_dir


class FunctionalSimilarity:
    def __init__(self, features, similarity):
        with open(features) as f:
            lines = f.readlines()
        self.features = np.array(
            [line.strip("\ufeff").strip("\n") for line in lines], dtype=str
        )

        with open(similarity) as f:
            lines = f.readlines()
        self.similarity = np.array([line.split(",") for line in lines], dtype=float)


class Dataset:
    def __init__(self, data, features, labels) -> None:
        with open(data) as f:
            lines = f.readlines()
        self.X = np.array([line.split(",") for line in lines], dtype=float)

        with open(features) as f:
            lines = f.readlines()
        self.features = np.array(
            [line.strip("\ufeff").strip("\n") for line in lines], dtype=str
        )

        with open(labels) as f:
            lines = f.readlines()
        self.labels = np.array(
            [line.strip("\ufeff").strip("\n") for line in lines], dtype=int
        )


functional_similarity = FunctionalSimilarity(
    features=data_dir + "/features.csv",
    similarity=data_dir + "/similarity.csv",
)

dataset = Dataset(
    data=data_dir + "/patient.csv",
    features=data_dir + "/features.csv",
    labels=data_dir + "/labels.csv",
)

# NOTE: FunctionalSimilarityとDatasetに共通するfeaturesを取得
features = dataset.features

X = dataset.X
y = dataset.labels
