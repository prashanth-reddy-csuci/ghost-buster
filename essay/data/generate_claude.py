import os
import tqdm
import openai
import math
import requests
import json

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

        if os.path.exists(f"claude/{idx}.txt") or not os.path.exists(f"human/{idx}.txt"):
            continue

        with open(f"human/{idx}.txt", "r") as f:
            doc = f.read()
            words = round_up_fifty(len(doc.split()))

        with open(f"gpt/prompts/{idx}.txt", "r") as f:
            prompt = f.read().strip()

        response = generate_claude(f"Write an essay in {words} words to the prompt: {prompt}", estimate_tokens(words))

        with open(f"claude/{idx}.txt", "w") as f:
            f.write(f"{response['completion'].strip()}")

    # print("Generating logprobs for GPT...")
    # for idx, file in enumerate(tqdm.tqdm(os.listdir("gpt"))):
    #     if not os.path.exists(f"gpt/{idx}.txt"):
    #         continue

    #     with open(f"gpt/{idx}.txt") as f:
    #         doc = f.read().strip()

    #     if not os.path.exists(f"gpt/logprobs/{idx}-davinci.txt"):
    #         write_logprobs(doc, f"gpt/logprobs/{idx}-davinci.txt", "davinci")

    #     if not os.path.exists(f"gpt/logprobs/{idx}-ada.txt"):
    #         write_logprobs(doc, f"gpt/logprobs/{idx}-ada.txt", "ada")
