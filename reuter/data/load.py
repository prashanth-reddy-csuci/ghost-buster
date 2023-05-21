import os
import tqdm
import numpy as np


def generate_dataset(featurize, split, author=None, verbose=True, base_dir="data"):
    data = []
    if author is None:
        authors = sorted(os.listdir(f"{base_dir}/human/train"))
    else:
        authors = [author]

    to_iter = tqdm.tqdm(authors) if verbose else authors

    for author in to_iter:
        for source in ["human", "gpt"]:
            for file in sorted(os.listdir(f"{base_dir}/{source}/{split}/{author}")):
                if file == "logprobs" or file == "headlines":
                    continue
                data.append(
                    featurize(f"{base_dir}/{source}/{split}/{author}/{file}"))

    return np.array(data)
