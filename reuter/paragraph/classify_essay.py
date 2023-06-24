import numpy as np
import dill as pickle
import tqdm
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from utils.featurize import normalize, t_featurize

from utils.symbolic import get_words, vec_functions, scalar_functions, get_all_logprobs

INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

best_features = open("best_features.txt").read().strip().split("\n")
n_documents = len(os.listdir("../data/paragraph/")) - 2


def get_label(file_name):
    directory = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)

    file_name_without_ext = os.path.splitext(base_name)[0]

    with open(f"../data/paragraph/labels/{file_name_without_ext}.txt") as f:
        label = f.read().strip()

    return 1 if label == "gpt" else 0


def generate_dataset(featurize, verbose=True):
    data = []
    to_iter = tqdm.tqdm(range(n_documents)) if verbose else range(n_documents)

    for file_idx in to_iter:
        data.append(featurize(f"../data/paragraph/{file_idx}.txt"))

    return np.array(data)


# Generate train data
print("Generating train data")
train_data = normalize(
    np.concatenate(
        [generate_dataset(t_featurize)] +
        [exp_to_data[e] for e in best_features], axis=1
    )
)
train_labels = generate_dataset(get_label)
# Generate test data
print("Generating test data")
exp_to_data = pickle.load(open("../../essay/paragraph/symbolic_data", "rb"))

data = []
y = []
for idx in tqdm.tqdm(range(1446)):
    data.append(t_featurize(f"../../essay/data/paragraph/{idx}.txt"))
    with open(f"../../essay/data/paragraph/labels/{idx}.txt") as f:
        source = f.read().strip()
        if source == "gpt":
            y.append(1)
        else:
            y.append(0)

data = np.array(data)

X = normalize(np.concatenate(
    [data] + [exp_to_data[e] for e in best_features],
    axis=1
))

print(f"Data Shape: {train_data.shape}")
model = LogisticRegression(C=10, penalty='l2', max_iter=1000)
model.fit(train_data, train_labels)

# Print accuracy, F1, and AUC
print(f"Accuracy: {accuracy_score(y, model.predict(X))}")
print(f"F1 Score: {f1_score(y, model.predict(X))}")
print(f"AUROC: {roc_auc_score(y, model.predict_proba(X)[:, 1])}")
