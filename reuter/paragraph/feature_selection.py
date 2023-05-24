import numpy as np
import dill as pickle
import tqdm
import torch
import os

from utils.score import k_fold_score
from utils.featurize import normalize
torch.random.manual_seed(0)

def generate_dataset(featurize, verbose=False):
    data = []
    to_iter = tqdm.tqdm(range(1222)) if verbose else range(1222)

    for file_idx in to_iter:
        data.append(featurize(f"../data/paragraph/{file_idx}.txt"))

    return np.array(data)

INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

np.random.seed(0)
indices = np.arange(1222)
np.random.shuffle(indices)
indices, _ = np.split(indices, [int(.8 * len(indices))])

normalized_exp_to_data = {}
for key in exp_to_data:
    normalized_exp_to_data[key] = normalize(exp_to_data[key])


def get_data(*exp):
    return np.concatenate(
        [normalized_exp_to_data[e] for e in exp],
        axis=1
    )

def get_label(file_name):
    directory = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)

    file_name_without_ext = os.path.splitext(base_name)[0]

    with open(f"../data/paragraph/labels/{file_name_without_ext}.txt") as f:
        label = f.read().strip()
    
    return 1 if label == "gpt" else 0

labels = generate_dataset(get_label)
print(len(labels), sum(labels))

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
