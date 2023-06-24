import os
import sys

authors = sorted(os.listdir("human/train"))

"""
Bad Documents:
clade/train/AaronPressman/0.txt
clade/train/DavidLawder/37.txt
clade/train/KirstinRidley/20.txt
clade/train/MartinWolk/20.txt
clade/train/SarahDavison/31.txt
clade/train/TheresePoletti/4.txt
clade/test/DavidLawder/3.txt
clade/test/EdnaFernandes/31.txt
clade/test/GrahamEarnshaw/8.txt
clade/test/LydiaZajc/22.txt
clade/test/PatriciaCommins/14.txt
clade/test/ToddNissen/2.txt
clade/test/ToddNissen/26.txt
clade/test/WilliamKazer/29.txt
"""

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


def filter_claude(split):
    for author in authors:
        for i in range(50):
            with open(f"claude/{split}/{author}/{i}.txt", "r") as f:
                doc = f.read().strip()
            if doc.strip().split(" ")[0].lower() == "here":
                with open(f"claude/{split}/{author}/{i}.txt", "w") as f:
                    f.write(doc[doc.index("\n")+1:].strip())
            else:
                print(f"clade/{split}/{author}/{i}.txt")


if __name__ == "__main__":
    filter_claude("train")
    filter_claude("test")
