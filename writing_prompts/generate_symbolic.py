from nltk.util import ngrams
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

import tqdm
import numpy as np
import tiktoken
import dill as pickle

from utils.featurize import *
from utils.n_gram import *

from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression

MAX_DEPTH = 2
OUTPUT_FILE = "symbolic_data"


def get_words(exp):
    """
    Splits up expression into words, to be individually processed
    """
    return exp.split(" ")


def backtrack_functions(prev="", depth=0):
    """
    Backtrack all possible features.
    """
    if depth >= MAX_DEPTH:
        return []

    all_funcs = []
    prev_word = get_words(prev)[-1]

    if prev != "":
        for func in scalar_functions:
            all_funcs.append(f"{prev} {func}")
    else:
        for func in scalar_functions:
            for vec in vectors:
                all_funcs.append(f"{vec} {func}")

    for comb in vec_combinations:
        if get_words(comb)[0] != prev_word:
            all_funcs += backtrack_functions(
                prev + " " * bool(prev) + comb,
                depth + 1
            )

    return all_funcs


def calc_features(source, idx, exp):
    exp_tokens = get_words(exp)
    curr = vectors[exp_tokens[0]](source, idx)

    for i in range(1, len(exp_tokens)):
        if exp_tokens[i] in vec_functions:
            next_vec = vectors[exp_tokens[i+1]](source, idx)
            curr = vec_functions[exp_tokens[i]](curr, next_vec)
        elif exp_tokens[i] in scalar_functions:
            return scalar_functions[exp_tokens[i]](curr)


def generate_dataset(featurize, verbose=False):
    data = []
    to_iter = tqdm.tqdm(range(1000)) if verbose else range(1000)

    for file_idx in to_iter:
        for source in ["human", "gpt"]:
            data.append(featurize(source, file_idx))

    return np.array(data)


enc = tiktoken.encoding_for_model("davinci")
tokenizer = enc.encode
vocab_size = enc.n_vocab

# We use the brown corpus to train the n-gram model
sentences = brown.sents()

print("Tokenizing corpus...")
tokenized_corpus = []
for sentence in tqdm.tqdm(sentences):
    tokens = tokenizer(' '.join(sentence))
    tokenized_corpus += tokens

print("\nTraining n-gram model...")
model = TrigramBackoff(tokenized_corpus)

davinci_logprobs, ada_logprobs, trigram_logprobs = {}, {}, {}

print("Loading logprobs into memory")
for i in tqdm.tqdm(range(1000)):
    # First, load human logprobs
    with open(f"data/human/{i}.txt", "r") as f:
        doc = f.read()
        doc = doc[doc.index("\n") + 1:]

    davinci_logprobs[("human", i)] = get_logprobs(
        f"data/human/logprobs/{i}-davinci.txt")
    ada_logprobs[("human", i)] = get_logprobs(
        f"data/human/logprobs/{i}-ada.txt")
    trigram_logprobs[("human", i)] = score_ngram(
        f"data/human/{i}.txt", model, tokenizer, strip_first=True)

    # Then, do the same for GPT
    with open(f"data/gpt/{i}.txt", "r") as f:
        doc = f.read()
        doc = doc[doc.index("\n") + 1:]

    davinci_logprobs[("gpt", i)] = get_logprobs(
        f"data/gpt/logprobs/{i}-davinci.txt")
    ada_logprobs[("gpt", i)] = get_logprobs(f"data/gpt/logprobs/{i}-ada.txt")
    trigram_logprobs[("gpt", i)] = score_ngram(
        f"data/gpt/{i}.txt", model, tokenizer, strip_first=True)

vec_functions = {
    "v-add": lambda a, b: a + b,
    "v-sub": lambda a, b: a - b,
    "v-mul": lambda a, b: a * b,
    "v-div": lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=(b != 0), casting='unsafe'),
    "v->": lambda a, b: a > b,
    "v-<": lambda a, b: a < b
}

scalar_functions = {
    "s-max": max,
    "s-min": min,
    "s-avg": lambda x: sum(x) / len(x),
    "s-avg-top-25": lambda x: sum(sorted(x, reverse=True)[:25]) / len(sorted(x, reverse=True)[:25]),
    "s-len": len,
    "s-var": np.var,
    "s-l2": np.linalg.norm
}

vectors = {
    "davinci-logprobs": lambda source, idx: davinci_logprobs[(source, idx)],
    "ada-logprobs": lambda source, idx: ada_logprobs[(source, idx)],
    "trigram-logprobs": lambda source, idx: trigram_logprobs[(source, idx)],
}

# Get vec_combinations

vec_combinations = []
vector_names = list(vectors.keys())

for vec1 in range(len(vectors)):
    for vec2 in range(vec1):
        for func in vec_functions:
            if func != "v-div":
                vec_combinations.append(
                    f"{vector_names[vec1]} {func} {vector_names[vec2]}")

for vec1 in vectors:
    for vec2 in vectors:
        if vec1 != vec2:
            vec_combinations.append(f"{vec1} v-div {vec2}")


all_funcs = backtrack_functions()

print(f"\nTotal # of Features: {len(all_funcs)}.")
print("Sampling 5 features:")
for i in range(5):
    print(all_funcs[np.random.randint(0, len(all_funcs))])

print("\nGenerating datasets...")
exp_to_data = {}

for exp in tqdm.tqdm(all_funcs):
    exp_to_data[exp] = generate_dataset(
        lambda source, idx: calc_features(source, idx, exp)
    ).reshape(-1, 1)

pickle.dump(exp_to_data, open(OUTPUT_FILE, "wb"))
