import os
import tqdm
import time
import math
import json
import requests

from utils.write_logprobs import write_logprobs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

authors = sorted(os.listdir("human/train"))
to_skip = {"logprobs", "headlines"}

API_KEY_PATH = "/home/nickatomlin/vivek/detect_ai/anthropic.config"
with open(API_KEY_PATH) as f:
    api_key = json.loads(f.read())["api_key"]


def round_up_fifty(x):
    return int(math.ceil(x / 250)) * 250


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


def generate_claude_articles(split, verbose=False):
    """
    Generate train/test split of GPT articles.
    """
    global authors
    authors = tqdm.tqdm(authors) if verbose else authors

    for author in authors:
        if not os.path.exists(f"claude/{split}/{author}"):
            os.mkdir(f"claude/{split}/{author}")

        # Create intermediary folders
        if not os.path.exists(f"claude/{split}/{author}/logprobs"):
            os.mkdir(f"claude/{split}/{author}/logprobs")

        for n, article in enumerate(tqdm.tqdm(sorted(os.listdir(f"human/{split}/{author}")))):
            # If the document already exists, skip it
            if os.path.exists(f"claude/{split}/{author}/{n}.txt") or article in to_skip:
                continue
                
            with open(f"human/{split}/{author}/{article}") as f:
                doc = f.read()
                words = round_up_fifty(len(doc.split(" ")))
            
            with open(f"gpt/{split}/{author}/headlines/{n}.txt", "r") as f:
                headline = f.read().strip()
            
            response = generate_claude(f"Write a news article in {words} words with the following headline: {headline}", estimate_tokens(words))

            # Write the article to file
            with open(f"claude/{split}/{author}/{n}.txt", "w") as f:
                f.write(f"{response['completion'].strip()}")


def generate_claude_logprobs(split, verbose=False):
    """
    Generate logprobs for train/test split of GPT articles.
    """
    global authors
    authors = tqdm.tqdm(authors) if verbose else authors

    for author in authors:
        for i in range(50):
            # If the document already exists, skip it
            with open(f"claude/{split}/{author}/{i}.txt") as f:
                text = f.read().strip()

            if not os.path.exists(f"claude/{split}/{author}/logprobs/{i}-ada.txt"):
                write_logprobs(
                    text, f"claude/{split}/{author}/logprobs/{i}-ada.txt", "ada"
                )

            if not os.path.exists(f"claude/{split}/{author}/logprobs/{i}-davinci.txt"):
                write_logprobs(
                    text, f"claude/{split}/{author}/logprobs/{i}-davinci.txt", "davinci"
                )


if __name__ == "__main__":
    # Create folders (if not present already)
    for author in authors:
        if not os.path.exists(f"claude/train/{author}"):
            os.mkdir(f"claude/train/{author}")
        if not os.path.exists(f"claude/test/{author}"):
            os.mkdir(f"claude/test/{author}")

    # Generate articles
    print("Generating Train Articles")
    generate_claude_articles(split="train", verbose=True)

    print("Generating Test Articles")
    generate_claude_articles(split="test", verbose=True)

    print("Generating logprobs")
    generate_claude_logprobs(split="train", verbose=True)
    generate_claude_logprobs(split="test", verbose=True)
