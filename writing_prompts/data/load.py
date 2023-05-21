import tqdm
import numpy as np

def generate_dataset(featurize, verbose=False):
    data = []
    to_iter = tqdm.tqdm(range(1000)) if verbose else range(1000)

    for file_idx in to_iter:
        for source in ["human", "gpt"]:
            data.append(featurize(f"data/{source}/{file_idx}.txt"))

    return np.array(data)