from utils.symbolic import generate_symbolic_data
from writing_prompts.data.utils import generate_dataset

generate_symbolic_data(
    generate_dataset,
    preprocess=lambda x: x[x.index("\n")+1:],
    output_file="symbolic_data",
    max_depth=2,
    verbose=True
)