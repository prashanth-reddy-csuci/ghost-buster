from utils.symbolic import generate_symbolic_data
from reuter.data.load import generate_dataset

generate_symbolic_data(
    lambda featurize: generate_dataset(featurize, "train"),
    output_file="symbolic_data",
    max_depth=3,
    verbose=True
)