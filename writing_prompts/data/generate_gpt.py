import os
import tqdm
import openai
import math

from utils import write_logprobs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def round_up_fifty(x):
    return int(math.ceil(x / 50.0)) * 50


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


if __name__ == "__main__":
    num_docs = 1000

    if not os.path.exists("gpt/logprobs"):
        os.mkdir("gpt/logprobs")

    print("Generating Articles...")
    for idx in tqdm.tqdm(range(num_docs)):
        if os.path.exists(f"gpt/{idx}.txt"):
            continue

        with open(f"human/{idx}.txt") as f:
            file = f.read().strip()
            prompt = file[:file.index("\n")].strip()
            response = file[file.index("\n") + 1:].strip()
            words = round_up_fifty(len(response.split(" ")))

        response = openai_backoff(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                            "content": f"Write a story in {words} words to the prompt: {prompt}",
                }
            ],
        )
        reply = response["choices"][0]["message"]["content"].strip()
        with open(f"gpt/{idx}.txt", "w") as f:
            f.write(f"{prompt}\n{reply}")

    print("Writing logprobs...")
    for idx in tqdm.tqdm(range(num_docs)):
        if os.path.exists(f"gpt/logprobs/{idx}.txt"):
            continue

        with open(f"gpt/{idx}.txt") as f:
            file = f.read().strip()
            doc = file[file.index("\n") + 1:].strip()

        write_logprobs(doc, f"gpt/logprobs/{idx}-davinci.txt", "davinci")
        write_logprobs(doc, f"gpt/logprobs/{idx}-ada.txt", "ada")
