import os
import sys

punct = ["...", ".", "?", "!", ",", ":", ";", "-", "}",
         "'", "\"", "'", "*", "n't", "'s", "]", ")"]
to_replace = [
    [" '", "'"],
    ["<newline> ", "\n"],
    ["<newline>", "\n"],
    ["''", "'"],
    ["`` ", "'"],
    [" [ ", " ["],
    [" ( ", " ("],
    [" { ", " {"],
]


def preprocess_text(text, title=False):
    text = text.strip() + " "
    if title:
        text = text[7:]

    for p in punct:
        text = text.replace(f" {p} ", f"{p} ")

    for s, t in to_replace:
        text = text.replace(s, t)

    return text.strip()


if __name__ == "__main__":
    num_docs = 1000

    titles = open("train.wp_source", "r").readlines()
    text = open("train.wp_target", "r").readlines()

    for idx in tqdm.tqdm(range(num_docs)):
        with open(f"human/{idx}.txt", "w") as f:
            f.write(
                f"{preprocess(titles[idx], True)}\n{preprocess(text[idx], False)}")
