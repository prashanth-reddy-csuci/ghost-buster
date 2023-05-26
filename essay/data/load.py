import tqdm
import numpy as np

def generate_dataset(featurize, verbose=False, base_dir=""):
    data = []
    to_iter = tqdm.tqdm(range(2685)) if verbose else range(2685)

    for file_idx in to_iter:
        for source in ["human", "gpt"]:
            data.append(featurize(f"{base_dir}data/{source}/{file_idx}.txt"))

    return np.array(data)