from dataset_predict import features, X, y
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from n_grams import f
from sklearn.cluster import KMeans
from path import root_dir


def selectCenterVectorIndex(vectors):
    center = np.mean(vectors, axis=0)

    distances = np.linalg.norm(vectors - center, axis=1)

    min_distance_index = np.argmin(distances)

    return min_distance_index


class Clusters:
    def __init__(self, cluster_labels):
        self.cluster_labels = cluster_labels
        self.indexes_by_cluster = {}
        for i, c in enumerate(cluster_labels):
            if c in self.indexes_by_cluster.keys():
                self.indexes_by_cluster[c].append(i)
            else:
                self.indexes_by_cluster[c] = [i]

    def sample_from_cluster(self, cluster_label, seed):
        random.seed(seed)
        return random.choice(self.indexes_by_cluster[cluster_label])


def main(cluster_labels, _X):
    clusters = Clusters(cluster_labels)
    label_list = []
    for i, label in enumerate(cluster_labels):
        if not label in label_list:
            label_list.append(label)
    label_list.sort()

    # NOTE: 0は除く
    label_list = label_list[1:]

    clf = LogisticRegression(random_state=0, max_iter=3000, multi_class="multinomial")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scaler = StandardScaler()

    selected_feature_indexes = []
    for i, label in enumerate(label_list):
        clusters_indexes = clusters.indexes_by_cluster[label]
        clusters_indexes.sort()
        index = selectCenterVectorIndex(_X[clusters_indexes])
        selected_feature_indexes.append(clusters_indexes[index])

    selected_feature_indexes.sort()
    print(len(selected_feature_indexes))
    scaled_X = scaler.fit_transform(X[:, selected_feature_indexes])
    scores = cross_val_score(clf, scaled_X, y, cv=cv)

    # 再現率
    recall = cross_val_score(clf, scaled_X, y, cv=cv, scoring="recall_macro")

    auc = cross_val_score(clf, scaled_X, y, cv=cv, scoring="roc_auc_ovr")

    print("mean: ", scores.mean())
    print("std: ", scores.std())

    print("recall: ", recall.mean())
    print("recall std: ", recall.std())

    print("auc: ", auc.mean())
    print("auc std: ", auc.std())


if __name__ == "__main__":
    _X = np.array(f)
    binary_vectors = np.zeros((len(features), 28), dtype=int)

    path = root_dir + "/results_gcn_hash"
    file_list = os.listdir(path)
    file_list.sort()
    for i, file_name in enumerate(file_list):
        with open(path + "/" + file_name, "r") as f:
            for j, line in enumerate(f):
                line = int(line.rstrip())
                binary_vectors[j][i] = line

    kmeans = KMeans(n_clusters=21, random_state=0).fit(binary_vectors)
    cluster_labels = kmeans.labels_
    cluster_labels = cluster_labels + 1

    for i, vector in enumerate(binary_vectors):
        # 全て0の場合
        if np.sum(vector) == 0:
            cluster_labels[i] = 0

    main(cluster_labels=cluster_labels, _X=_X)
