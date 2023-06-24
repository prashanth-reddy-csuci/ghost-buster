import tqdm
import numpy as np
import os


def generate_dataset(featurize, verbose=False, base_dir="", split=range(2000)):
    data = []
    to_iter = tqdm.tqdm(split) if verbose else split

    for file_idx in to_iter:
        if file_idx % 2 == 0:
            source, idx = "human", file_idx // 2
        else:
            source, idx = "gpt", file_idx // 2
        data.append(featurize(f"{base_dir}data/{source}/{idx}.txt"))

    return np.array(data)


def generate_dataset_claude(featurize, verbose=False, base_dir="", split=range(2000)):
    data = []
    to_iter = tqdm.tqdm(split) if verbose else split

    for file_idx in to_iter:
        if file_idx % 2 == 0:
            source, idx = "human", file_idx // 2
        else:
            source, idx = "claude", file_idx // 2
        data.append(featurize(f"{base_dir}data/{source}/{idx}.txt"))

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
