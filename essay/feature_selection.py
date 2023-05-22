import numpy as np
import dill as pickle
import tqdm
import torch

from utils.score import k_fold_score
from utils.featurize import normalize
from essay.data.load import generate_dataset
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


labels = generate_dataset(lambda file: 1 if "gpt" in file else 0)
assert sum(labels) == len(labels) // 2

val_exp = list(exp_to_data.keys())
curr = 0
best_features = []
i = 0

while val_exp:
    best_score, best_exp = -1, ""

    for exp in tqdm.tqdm(val_exp):
        score = k_fold_score(get_data(*best_features, exp), labels, k=5, indices=indices)

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
