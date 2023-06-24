import numpy as np
import dill as pickle
import tqdm
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from utils.featurize import normalize, t_featurize


def generate_dataset(featurize, verbose=False):
    data = []
    to_iter = tqdm.tqdm(range(1798)) if verbose else range(1798)

    for file_idx in to_iter:
        data.append(featurize(f"../data/paragraph/{file_idx}.txt"))

    return np.array(data)

INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

np.random.seed(0)
indices = np.arange(1798)
np.random.shuffle(indices)
train_indices, test_indices = np.split(indices, [int(.8 * len(indices))])

best_features = open("best_features.txt").read().strip().split("\n")

def get_label(file_name):
    directory = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)

    file_name_without_ext = os.path.splitext(base_name)[0]

    with open(f"../data/paragraph/labels/{file_name_without_ext}.txt") as f:
        label = f.read().strip()
    
    return 1 if label == "gpt" else 0

data = generate_dataset(t_featurize, verbose=True)

X = normalize(np.concatenate(
    [exp_to_data[e] for e in best_features] + [data],
    axis=1
))
y = generate_dataset(get_label)


print(f"Data Shape: {X.shape}")
model = LogisticRegression(C=10, penalty='l2', max_iter=1000)
model.fit(X[train_indices], y[train_indices])


# Print accuracy
print(
    f"Accuracy: {accuracy_score(y[test_indices], model.predict(X[test_indices]))}")

# Print f1 score
print(f"F1 Score: {f1_score(y[test_indices], model.predict(X[test_indices]))}")

# Print AUROC
print(
    f"AUROC: {roc_auc_score(y[test_indices], model.predict_proba(X[test_indices])[:, 1])}")
