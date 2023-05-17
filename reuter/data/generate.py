import os
import tqdm
import time
import openai
import math

from utils import write_logprobs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

authors = sorted(os.listdir("human/train"))
to_skip = {"logprobs", "headlines"}


def round_up_fifty(x):
    return int(math.ceil(x / 50.0)) * 50


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def generate_gpt_articles(split, verbose=False):
    """
    Generate train/test split of GPT articles.
    """
    global authors
    authors = tqdm.tqdm(authors) if verbose else authors

    for author in authors:
        # Create intermediary folders
        if not os.path.exists(f"gpt/{split}/{author}/logprobs"):
            os.mkdir(f"gpt/{split}/{author}/logprobs")
        if not os.path.exists(f"gpt/{split}/{author}/headlines"):
            os.mkdir(f"gpt/{split}/{author}/headlines")

        for n, article in enumerate(sorted(os.listdir(f"human/{split}/{author}"))):
            # If the document already exists, skip it
            if os.path.exists(f"gpt/{split}/{author}/{n}.txt") or article in to_skip:
                continue

            # Generate a headline for the article
            with open(f"human/{split}/{author}/{article}") as f:
                doc = f.read()
                response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Create a headline for the following news article: {doc}",
                        }
                    ],
                )
            headline = response["choices"][0]["message"]["content"].strip()

            # Write the headline to file
            with open(f"gpt/{split}/{author}/headlines/{n}.txt", "w") as f:
                f.write(headline)

            # Generate the article
            length = round_up_fifty(len(doc.split(" ")))
            response = openai_backoff(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"Write a news article in {length} words with the following headline: {headline}.",
                    }
                ],
            )
            article = response["choices"][0]["message"]["content"].strip()
            article = article.replace("\n\n", "\n").strip()

            # Write the article to file
            with open(f"gpt/{split}/{author}/{n}.txt", "w") as f:
                f.write(article)


if __name__ == "__main__":
    # Create folders (if not present already)
    for author in authors:
        if not os.path.exists(f"gpt/train/{author}"):
            os.mkdir(f"gpt/train/{author}")
        if not os.path.exists(f"gpt/test/{author}"):
            os.mkdir(f"gpt/test/{author}")

    # Generate articles
    print("Generating Train Articles")
    generate_gpt_articles(split="train", verbose=True)
    
    print("Generating Test Articles")
    generate_gpt_articles(split="test", verbose=True)
