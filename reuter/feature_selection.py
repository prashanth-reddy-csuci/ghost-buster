import numpy as np
import dill as pickle
import tqdm
import torch

from utils.score import k_fold_score
from utils.featurize import normalize, select_features
from reuter.data.load import generate_dataset
torch.random.manual_seed(0)


INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

labels = generate_dataset(lambda file: 1 if "gpt" in file else 0)
assert sum(labels) == len(labels) // 2

best_features = select_features(
    exp_to_data, labels, verbose=True, normalize=True)

print(f"Final Score: {curr}, Best Features:")
for f in best_features:
    print(f)

with open("best_features.txt", "w") as f:
    for feat in best_features:
        f.write(feat + "\n")
