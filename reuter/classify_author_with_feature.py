import numpy as np
import dill as pickle
import tqdm
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from utils.featurize import normalize, t_featurize, select_features
from reuter.data.load import generate_dataset

from utils.symbolic import *
from utils.score import k_fold_score

accuracies, f1_scores, aucs = {}, {}, {}

authors = sorted(os.listdir("data/human/train"))
best_features_global = open("best_features.txt").read().strip().split("\n")

all_feautres = backtrack_functions(max_depth=2)


def get_exp_featurize(vector_map, best_features):
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

    return exp_featurize


def get_train_split(author):
    davinci_logprobs, ada_logprobs, trigram_logprobs, unigram_logprobs = get_all_logprobs(
        lambda featurize: generate_dataset(
            featurize, "train", author=author, verbose=False),
        tokenizer=tokenizer, trigram=trigram,
        verbose=False
    )
    vector_map = {
        "davinci-logprobs": lambda file: davinci_logprobs[file],
        "ada-logprobs": lambda file: ada_logprobs[file],
        "trigram-logprobs": lambda file: trigram_logprobs[file],
        "unigram-logprobs": lambda file: unigram_logprobs[file]
    }

    exp_to_data = {}
    for exp in all_feautres:
        exp_to_data[exp] = generate_dataset(
            get_exp_featurize(vector_map, [exp]), "train", author=author, verbose=False
        )
    labels = generate_dataset(
        lambda file: 1 if "gpt" in file else 0, "train", author=author, verbose=False)

    best_features = select_features(
        exp_to_data, labels, verbose=False, to_normalize=True) + best_features_global

    data = normalize(
        np.concatenate(
            [generate_dataset(t_featurize, "train", author=author, verbose=False),
             generate_dataset(get_exp_featurize(vector_map, best_features), "train", author=author, verbose=False)],
            axis=1
        )
    )

    assert len(data) == len(labels) and sum(labels) == len(labels) // 2

    return data, labels, best_features


def get_test_split(author, best_features):
    davinci_logprobs, ada_logprobs, trigram_logprobs, unigram_logprobs = get_all_logprobs(
        lambda featurize: generate_dataset(
            featurize, "test", author=author, verbose=False),
        tokenizer=tokenizer, trigram=trigram,
        verbose=False
    )
    vector_map = {
        "davinci-logprobs": lambda file: davinci_logprobs[file],
        "ada-logprobs": lambda file: ada_logprobs[file],
        "trigram-logprobs": lambda file: trigram_logprobs[file],
        "unigram-logprobs": lambda file: unigram_logprobs[file]
    }

    data = normalize(
        np.concatenate(
            [generate_dataset(t_featurize, "test", author=author, verbose=False),
             generate_dataset(get_exp_featurize(vector_map, best_features), "test", author=author, verbose=False)],
            axis=1
        )
    )
    labels = generate_dataset(
        lambda file: 1 if "gpt" in file else 0, "test", author=author, verbose=False)
    print(data.shape)

    assert len(data) == len(labels) and sum(labels) == len(labels) // 2

    return data, labels


trigram, tokenizer = train_trigram(
    verbose=True, return_tokenizer=True)

for author in tqdm.tqdm(authors):
    train_data, train_labels, best_features = get_train_split(author)
    test_data, test_labels = get_test_split(author, best_features)

    model = LogisticRegression(random_state=0, max_iter=1000).fit(
        train_data, train_labels)

    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data)[:, 1]

    accuracies[author] = accuracy_score(test_labels, predictions)
    f1_scores[author] = f1_score(test_labels, predictions)
    aucs[author] = roc_auc_score(test_labels, probabilities)

# Print average accuracy, f1, and auroc
print(f"Average Accuracy: {np.mean(list(accuracies.values()))}")
print(f"Average F1 Score: {np.mean(list(f1_scores.values()))}")
print(f"Average AUROC: {np.mean(list(aucs.values()))}")
