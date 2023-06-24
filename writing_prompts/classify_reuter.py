import numpy as np
import dill as pickle
import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from utils.featurize import normalize, t_featurize
from reuter.data.load import generate_dataset

INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

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
y = np.array([0, 1] * 1000)

print(f"Data Shape: {X.shape}")
model = LogisticRegression(C=10, penalty='l2', max_iter=1000)
model.fit(X, y)

exp_to_data = pickle.load(open("../reuter/symbolic_data", "rb"))

# Generate train data
print("Generating test data")
test_data = normalize(np.concatenate(
    [exp_to_data[e] for e in best_features] +
    [generate_dataset(t_featurize, "train", base_dir="../reuter/data")], axis=1
))

test_labels = generate_dataset(
    lambda file: 1 if "gpt" in file else 0, "train",
    base_dir="../reuter/data")


# Print accuracy
print(f"Accuracy: {accuracy_score(test_labels, model.predict(test_data))}")

# Print f1 score
print(f"F1 Score: {f1_score(test_labels, model.predict(test_data))}")

# Print AUROC
print(
    f"AUROC: {roc_auc_score(test_labels, model.predict_proba(test_data)[:, 1])}")
