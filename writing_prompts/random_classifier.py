import numpy as np
import dill as pickle
import tqdm

from sklearn.linear_model import LogisticRegression
from utils.featurize import normalize, t_featurize

NUM_FEATURES = 7
NUM_TRIALS = 100
INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

with open("data/train.txt") as f:
    train_indices = list(map(int, f.read().strip().split("\n")))
with open("data/test.txt") as f:
    test_indices = list(map(int, f.read().strip().split("\n")))

np.random.seed(100)

score = 0
for _ in range(NUM_TRIALS):
    features = np.random.choice(
        list(exp_to_data.keys()), size=NUM_FEATURES, replace=False)

    X = normalize(np.concatenate(
        [exp_to_data[e] for e in features],
        axis=1
    ))
    y = np.array([1, 0] * 1000)

    model = LogisticRegression(C=10, penalty='l2', max_iter=1000)
    model.fit(X[train_indices], y[train_indices])

    score += model.score(X[test_indices], y[test_indices])

print(f"Average Test Data Score: {score / NUM_TRIALS}")
