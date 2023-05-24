import os
import tqdm
import openai
import math
import numpy as np
import tiktoken

from utils import write_logprobs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

enc = tiktoken.encoding_for_model("davinci")
tokenizer = enc.encode
vocab_size = enc.n_vocab

max_context_size = 4000


def get_len(text):
    return len(tokenizer(text))


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


if __name__ == "__main__":
    np.random.seed(0)

    for file_idx in tqdm.tqdm(range(100)):
        if os.path.exists(f"paragraph/{file_idx}.txt"):
            continue

        with open(f"human/{file_idx}.txt") as f:
            doc = f.read()

        paragraphs = []
        length = 0
        p = np.random.uniform(0, 1)

        for para in doc.split("\n"):
            if not para.strip():
                continue

            if length + get_len(para) > max_context_size:
                break

            paragraphs.append(para.strip())
            length += get_len(para)

        if not paragraphs:
            print(f"Skipping {file_idx}")
            continue

        new_doc = paragraphs[:]
        sources = ["human"] * len(paragraphs)

        for i in range(1, len(paragraphs)-1):
            if np.random.uniform(0, 1) <= p:
                continue

            prompt = "\n".join(paragraphs[:i])

            new_para = ""
            tries = 0
            done = False
            while new_para == "":
                if tries > 3:
                    done = True
                    break
                response = openai_backoff(
                    model="text-davinci-003",
                    prompt=prompt + "\n\n",
                    suffix="\n\n" + "\n".join(paragraphs[i+1:]),
                    max_tokens=max_context_size -
                    length + get_len(paragraphs[i]),
                )
                new_para = response.choices[0]["text"]
                new_para = new_para.replace("\n", " ").strip()
                tries += 1
            if done:
                continue

            new_doc[i] = new_para
            sources[i] = "gpt"

        with open(f"paragraph/{file_idx}.txt", "w") as f:
            f.write("\n".join(new_doc))

        with open(f"paragraph/labels/{file_idx}.txt", "w") as f:
            f.write("\n".join(sources))

        with open(f"paragraph/labels/{file_idx}-prob.txt", "w") as f:
            f.write(str(p))

    for file_idx in tqdm.tqdm(range(1798)):
        with open(f"paragraph/{file_idx}.txt") as f:
            doc = f.read().strip()

        if not os.path.exists(f"paragraph/logprobs/{file_idx}-davinci.txt"):
            write_logprobs(
                doc, f"paragraph/logprobs/{file_idx}-davinci.txt", "davinci")

        if not os.path.exists(f"paragraph/logprobs/{file_idx}-ada.txt"):
            write_logprobs(
                doc, f"paragraph/logprobs/{file_idx}-ada.txt", "ada")
