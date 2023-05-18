import numpy as np
import dill as pickle
import tqdm
import torch

from sklearn.linear_model import LogisticRegression
from utils.featurize import normalize
from torch.utils.data import random_split
torch.random.manual_seed(0)


INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

with open("data/train.txt") as f:
    indices = list(map(int, f.read().strip().split("\n")))

normalized_exp_to_data = {}
for key in exp_to_data:
    normalized_exp_to_data[key] = normalize(exp_to_data[key])


def get_data(*exp):
    return np.concatenate(
        [normalized_exp_to_data[e] for e in exp],
        axis=1
    )


def k_fold_score(X, k=8):
    k = 8
    k_split = random_split(indices, [len(indices) // k] * k)

    score_sum = 0
    for i in range(k):
        train = np.concatenate([np.array(k_split[j])
                               for j in range(k) if i != j])
        model = LogisticRegression(C=10, penalty='l2', max_iter=1000)
        model.fit(X[train], labels[train])
        score_sum += model.score(X[k_split[i]], labels[k_split[i]])

    return round(score_sum / k, 3)


labels = np.array([1, 0] * 1000)

val_exp = list(exp_to_data.keys())
curr = 0
best_features = []
i = 0

while val_exp:
    best_score, best_exp = -1, ""

    for exp in tqdm.tqdm(val_exp):
        score = k_fold_score(get_data(*best_features, exp), k=5)

        if score > best_score:
            best_score = score
            best_exp = exp

    print(f"Iteration {i}, Current Score: {curr}, \
          Best Feature: {best_exp}, New Score: {best_score}")

    if best_score <= curr:
        break
    else:
        best_features.append(best_exp)
        val_exp.remove(best_exp)
        curr = best_score

    i += 1

print(f"Final Score: {curr}, Best Features:")
for f in best_features:
    print(f)

with open("best_features.txt", "w") as f:
    for feat in best_features:
        f.write(feat + "\n")
