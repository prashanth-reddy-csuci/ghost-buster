tot = 0

for i in range(1000):
    with open(f"human/{i}.txt") as f:
        doc = f.read().strip()
        doc = doc[doc.index("\n") + 1:]
        doc = doc.replace("\n\n", "\n")
        tot += len(doc.split("\n"))

print(tot)