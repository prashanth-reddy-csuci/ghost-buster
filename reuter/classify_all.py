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
t_test_data = generate_dataset(t_featurize, "test")

davinci_logprobs, ada_logprobs, trigram_logprobs = get_all_logprobs(
    lambda featurize: generate_dataset(featurize, "test"))

print("Loading logprobs into memory")

vector_map = {
    "davinci-logprobs": lambda file: davinci_logprobs[file],
    "ada-logprobs": lambda file: ada_logprobs[file],
    "trigram-logprobs": lambda file: trigram_logprobs[file]
}


def calc_features(file, exp):
    exp_tokens = get_words(exp)
    curr = vector_map[exp_tokens[0]](file)

    for i in range(1, len(exp_tokens)):
        if exp_tokens[i] in vec_functions:
            next_vec = vector_map[exp_tokens[i+1]](file)
            curr = vec_functions[exp_tokens[i]](curr, next_vec)
        elif exp_tokens[i] in scalar_functions:
            return scalar_functions[exp_tokens[i]](curr)


def exp_featurize(file):
    return np.array([calc_features(file, exp) for exp in best_features])


exp_test_data = generate_dataset(exp_featurize, "test")

test_data = normalize(
    np.concatenate(
        (t_test_data, exp_test_data),
        axis=1
    )
)
test_labels = generate_dataset(lambda file: 1 if "gpt" in file else 0, "test")

print(f"Data Shape: {train_data.shape}")
model = LogisticRegression(C=10, penalty='l2', max_iter=1000)
model.fit(train_data, train_labels)

# Print accuracy, F1, and AUC
print(f"Accuracy: {accuracy_score(test_labels, model.predict(test_data))}")
print(f"F1 Score: {f1_score(test_labels, model.predict(test_data))}")
print(f"AUROC: {roc_auc_score(test_labels, model.predict_proba(test_data)[:, 1])}")
