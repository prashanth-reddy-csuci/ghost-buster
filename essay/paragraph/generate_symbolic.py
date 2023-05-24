from utils.symbolic import generate_symbolic_data
import numpy as np
import tqdm


def generate_dataset(featurize, verbose=False):
    data = []
    to_iter = tqdm.tqdm(range(1446)) if verbose else range(1446)

    for file_idx in to_iter:
        data.append(featurize(f"../data/paragraph/{file_idx}.txt"))

    return np.array(data)


generate_symbolic_data(
    generate_dataset,
    output_file="symbolic_data",
    max_depth=3,
    verbose=True
)
