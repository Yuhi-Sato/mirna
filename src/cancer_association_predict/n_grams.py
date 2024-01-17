import numpy as np
from pathlib import Path

rootpath = str(Path(__file__).resolve().parent.parent.parent)

# NOTE: 塩基配列の配列
sequences = []
with open(rootpath + "/data/sequence.csv", "r") as f:
    for line in f:
        line = line.rstrip().split(",")
        sequences.append(line[0])

# NOTE: (i+1)-gramの総数
N = [4, 16, 64, 256]

# NOTE: 濃度因子の計算
C = [0, 0, 0, 0]
for i in range(4):
    C[i] = N[i] / sum(N)

# NOTE: 塩基
base = ["A", "U", "G", "C"]

# NOTE: n-gramの作成
unigrams = [a for a in base]
bigrams = [a + b for a in base for b in base]
trigrams = [a + b + c for a in base for b in base for c in base]
tetragrams = [a + b + c + d for a in base for b in base for c in base for d in base]

n_grams = unigrams + bigrams + trigrams + tetragrams

f = []
for sequence in sequences:
    t = [0] * 340
    T = [0] * 4
    for j in range(340):
        n_gram = n_grams[j]
        i = len(n_gram)
        for k in range(len(sequence) - i + 1):
            if n_gram == sequence[k : k + i]:
                t[j] += 1
                T[i - 1] += 1

    v = []
    for j in range(340):
        n_gram = n_grams[j]
        i = len(n_gram)
        v.append((t[j] / T[i - 1]) * C[i - 1])
    f.append(v)

f = np.array(f)
