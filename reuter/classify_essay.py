import numpy as np
import dill as pickle
import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from utils.featurize import normalize, t_featurize
from reuter.data.load import generate_dataset

from utils.symbolic import get_words, vec_functions, scalar_functions, get_all_logprobs

INPUT_FILE = "symbolic_data"
exp_to_data = pickle.load(open(INPUT_FILE, "rb"))

best_features = open("best_features.txt").read().strip().split("\n")


# Generate train data
print("Generating train data")
train_data = normalize(
    np.concatenate(
        [generate_dataset(t_featurize, "train")] +
        [exp_to_data[e] for e in best_features], axis=1
    )
)
train_labels = generate_dataset(
    lambda file: 1 if "gpt" in file else 0, "train")
assert sum(train_labels) == len(
    train_labels) // 2 and len(train_labels) == len(train_data)

# Generate test data
print("Generating test data")
exp_to_data = pickle.load(open("../essay/symbolic_data", "rb"))

data = []
for idx in tqdm.tqdm(range(2685)):
    for source in ["human", "gpt"]:
        data.append(t_featurize(f"../essay/data/{source}/{idx}.txt"))
data = np.array(data)

X = normalize(np.concatenate(
     [data] + [exp_to_data[e] for e in best_features],
    axis=1
))
y = np.array([0, 1] * 2685)

print(f"Data Shape: {train_data.shape}")
model = LogisticRegression(C=10, penalty='l2', max_iter=1000)
model.fit(train_data, train_labels)

# Print accuracy, F1, and AUC
print(f"Accuracy: {accuracy_score(y, model.predict(X))}")
print(f"F1 Score: {f1_score(y, model.predict(X))}")
print(f"AUROC: {roc_auc_score(y, model.predict_proba(X)[:, 1])}")
