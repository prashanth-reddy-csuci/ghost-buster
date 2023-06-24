from utils.symbolic import generate_symbolic_data
from utils.load import get_generate_dataset

generate_dataset = get_generate_dataset(
    base_dir="data",
)

generate_symbolic_data(
    generate_dataset,
    output_file="symbolic_data",
    max_depth=3,
    verbose=True
)
