import os
import tqdm
import numpy as np





def generate_dataset(featurize, split, author=None, verbose=True):
    data = []
    if author is not None:
        authors = sorted(os.listdir("data/human/train"))
    else:
        authors = [author]
        
    to_iter = tqdm.tqdm(authors) if verbose else authors

    for author in to_iter:
        for source in ["human", "gpt"]:
            for file in sorted(os.listdir(f"data/{source}/{split}/{author}")):
                if file == "logprobs" or file == "headlines":
                    continue
                data.append(
                    featurize(f"data/{source}/{split}/{author}/{file}"))

    return np.array(data)
