from utils.symbolic import generate_symbolic_data
from essay.data.load import generate_dataset

generate_symbolic_data(
    generate_dataset,
    output_file="symbolic_data",
    max_depth=2,
    verbose=True
)