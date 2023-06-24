import os
import tqdm

from utils.write_logprobs import write_logprobs

authors = sorted(os.listdir("human_author"))

if __name__ == "__main__":
    for author in tqdm.tqdm(authors):
        if not os.path.exists(f"human_author/{author}/logprobs"):
            os.mkdir(f"human_author/{author}/logprobs")

        for i in range(100):
            with open(f"human_author/{author}/{i}.txt") as f:
                doc = f.read().strip()

            if not os.path.exists(f"human_author/{author}/logprobs/{i}-ada.txt"):
                write_logprobs(
                    doc, f"human_author/{author}/logprobs/{i}-ada.txt", "ada")

            if not os.path.exists(f"human_author/{author}/logprobs/{i}-davinci.txt"):
                write_logprobs(
                    doc, f"human_author/{author}/logprobs/{i}-davinci.txt", "davinci")
