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
    if not os.path.exists("gpt/logprobs"):
        os.mkdir("gpt/logprobs")

    if not os.path.exists("gpt/prompts"):
        os.mkdir("gpt/prompts")

    print("Generating Articles...")
    for idx, file in enumerate(tqdm.tqdm(os.listdir("human"))):
        if file == "logprobs":
            continue

        if os.path.exists(f"gpt/{idx}.txt") or not os.path.exists(f"human/{idx}.txt"):
            continue

        with open(f"human/{idx}.txt") as f:
            doc = f.read()
            words = round_up_fifty(len(doc.split()))
            response = openai_backoff(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"Given the following essay, write a prompt for it.\n\n{' '.join(doc.split()[:1000])}",
                    }
                ],
            )
        prompt = response["choices"][0]["message"]["content"].strip()
        prompt = prompt.replace("Prompt:", "").strip()

        with open(f"gpt/prompts/{idx}.txt", "w") as f:
            f.write(prompt)

        response = openai_backoff(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                            "content": f"Write an essay in {words} words to the prompt: {prompt}",
                }
            ],
        )
        reply = response["choices"][0]["message"]["content"].strip()
        reply = reply.replace("\n\n", "\n")

        with open(f"gpt/{idx}.txt", "w") as f:
            f.write(f"{reply}")

    print("Generating logprobs for GPT...")
    for idx, file in enumerate(tqdm.tqdm(os.listdir("gpt"))):
        if file == "logprobs" or file == "prompts" or not os.path.exists(f"gpt/{idx}.txt"):
            continue

        with open(f"gpt/{idx}.txt") as f:
            doc = f.read().strip()

        if not os.path.exists(f"gpt/logprobs/{idx}-davinci.txt"):
            write_logprobs(doc, f"gpt/logprobs/{idx}-davinci.txt", "davinci")

        if not os.path.exists(f"gpt/logprobs/{idx}-ada.txt"):
            write_logprobs(doc, f"gpt/logprobs/{idx}-ada.txt", "ada")

    # Do the same for human
    print("Generating logprobs for human...")
    for idx, file in enumerate(tqdm.tqdm(os.listdir("human"))):
        if file == "logprobs" or not os.path.exists(f"human/{idx}.txt"):
            continue

        with open(f"human/{idx}.txt", "r") as f:
            doc = f.read().strip()

        if not os.path.exists(f"human/logprobs/{idx}-davinci.txt"):
            write_logprobs(doc, f"human/logprobs/{idx}-davinci.txt", "davinci")

        if not os.path.exists(f"human/logprobs/{idx}-ada.txt"):
            write_logprobs(doc, f"human/logprobs/{idx}-ada.txt", "ada")
