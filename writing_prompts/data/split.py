import numpy as np

np.random.seed(100)

indices = np.arange(2000)
np.random.shuffle(indices)
train, test = np.split(indices, [1600])

with open("train.txt", "w") as f:
    for i in train:
        f.write(str(i) + "\n")

with open("test.txt", "w") as f:
    for i in test:
        f.write(str(i) + "\n")
