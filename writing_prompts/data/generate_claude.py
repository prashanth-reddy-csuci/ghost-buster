import os
import tqdm
import openai
import math
import json
import requests

from utils import write_logprobs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

API_KEY_PATH = "/home/nickatomlin/vivek/detect_ai/anthropic.config"
with open(API_KEY_PATH) as f:
    api_key = json.loads(f.read())["api_key"]


def round_up_fifty(x):
    return int(math.ceil(x / 50.0)) * 50


def estimate_tokens(x):
    return round_up_fifty(x) * 4


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_claude(prompt, max_tokens):
    headers = {"x-api-key": api_key,
               "Content-Type": 'application/json'}
    prompt = f"\n\nHuman: {prompt.strip()}\n\nAssistant:"
    data = {"prompt": prompt,
            "model": 'claude-v1', "max_tokens_to_sample": max_tokens}
    r = requests.post('https://api.anthropic.com/v1/complete',
                      headers=headers, json=data)
    if r.status_code != 200:
        print(r.json())
        raise Exception("API Error")
    return r.json()


if __name__ == "__main__":
    num_docs = 1000

    if not os.path.exists("claude/logprobs"):
        os.mkdir("claude/logprobs")

    print("Generating Articles...")
    for idx in tqdm.tqdm(range(num_docs)):
        if os.path.exists(f"claude/{idx}.txt"):
            continue

        with open(f"human/{idx}.txt") as f:
            response = f.read().strip()
            words = round_up_fifty(len(response.split(" ")))

        with open(f"human/prompts/{idx}.txt") as f:
            prompt = f.read().strip()

        response = generate_claude(
            f"Write a story in {words} to the prompt {prompt}", estimate_tokens(words))

        with open(f"claude/{idx}.txt", "w") as f:
            f.write(f"{response['completion'].strip()}")

    print("Writing logprobs...")
    for idx in tqdm.tqdm(range(num_docs)):
        if os.path.exists(f"claude/logprobs/{idx}.txt"):
            continue

        with open(f"claude/{idx}.txt") as f:
            file = f.read().strip()
            doc = file[file.index("\n") + 1:].strip()

        write_logprobs(doc, f"claude/logprobs/{idx}-davinci.txt", "davinci")
        write_logprobs(doc, f"claude/logprobs/{idx}-ada.txt", "ada")
