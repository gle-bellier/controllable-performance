import pickle
from os import walk
import numpy as np


def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)

        except EOFError:
            pass


PATH = "data/contours/expressive/val.pickle"

f0 = lo = np.array([])

for c in read_from_pickle(PATH):
    f0 = np.concatenate((f0, c["f0"]))
    lo = np.concatenate((lo, c["lo"]))
print(f"Length: {len(f0)/100}s", )
print(f"Fundamental frequency: min {f0.min()} | max {f0.max()}")
print(f"Fundamental frequency: mean {f0.mean()} | std {f0.std()}")

print(f"Loudness: min {lo.min()} | max {lo.max()}")
print(f"Loudness: mean {lo.mean()} | std {lo.std()}")
