import tqdm
import numpy as np
import os


def generate_dataset(featurize, verbose=False):
    data = []
    to_iter = tqdm.tqdm(range(1000)) if verbose else range(1000)

    for file_idx in to_iter:
        for source in ["human", "gpt"]:
            data.append(featurize(f"data/{source}/{file_idx}.txt"))

    return np.array(data)


def generate_dataset_author(featurize, author, verbose=False, seed=None):
    if seed is None:
        author_id = sorted(os.listdir("data/human_author")).index(author)
        np.random.seed(author_id) 
    else:
        np.random.seed(seed)

    data = []
    to_iter = tqdm.tqdm(range(100)) if verbose else range(100)
    gpt_indices = np.random.choice(np.arange(1000), size=100, replace=False)

    for i in to_iter:
        data.append(featurize(f"data/human_author/{author}/{i}.txt"))
        data.append(featurize(f"data/gpt/{gpt_indices[i]}.txt"))

    return data