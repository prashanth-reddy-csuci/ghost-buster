import numpy as np
import dill as pickle
import tqdm

from sklearn.linear_model import LogisticRegression
from utils.featurize import normalize, t_featurize

INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

with open("data/train.txt") as f:
    train_indices = list(map(int, f.read().strip().split("\n")))
with open("data/test.txt") as f:
    test_indices = list(map(int, f.read().strip().split("\n")))

best_features = open("best_features.txt").read().strip().split("\n")

data = []
for idx in tqdm.tqdm(range(1000)):
    for source in ["human", "gpt"]:
        data.append(t_featurize(f"data/{source}/{idx}.txt"))
data = np.array(data)

X = normalize(np.concatenate(
    [exp_to_data[e] for e in best_features] + [data],
    axis=1
))
y = np.array([1, 0] * 1000)

print(f"Data Shape: {X.shape}")
model = LogisticRegression(C=10, penalty='l2', max_iter=1000)
model.fit(X[train_indices], y[train_indices])
print(f"Test Data Score: {model.score(X[test_indices], y[test_indices])}")
