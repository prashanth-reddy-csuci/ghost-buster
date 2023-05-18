import os
import sys

authors = sorted(os.listdir("human/train"))


def filter_split(split):
    for author in authors:
        for i in range(50):
            with open(f"gpt/{split}/{author}/headlines/{i}.txt") as f:
                headline = f.read().strip()

            with open(f"gpt/{split}/{author}/{i}.txt", "r") as f:
                doc = f.read().strip()

            first_line = doc[:len(headline)]
            if first_line == headline:
                doc = doc[len(headline):].strip()
                with open(f"gpt/{split}/{author}/{i}.txt", "w") as f:
                    f.write(doc)

            if doc[0] == ".":
                doc = doc[1:].strip()
                with open(f"gpt/{split}/{author}/{i}.txt", "w") as f:
                    f.write(doc)


if __name__ == "__main__":
    filter_split("train")
    filter_split("test")
