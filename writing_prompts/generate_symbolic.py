from utils.symbolic import generate_symbolic_data
from writing_prompts.data.load import generate_dataset

generate_symbolic_data(
    generate_dataset,
    preprocess=lambda x: x,
    output_file="symbolic_data",
    max_depth=3,
    verbose=True
)