import pickle
from os import walk
import numpy as np
from random import shuffle


def export(data: dict, path: str) -> None:
    """Export data into pickle file
    Args:
        data (dict): data dictionary
        path (str): path to the file
    """
    with open(path, "ab+") as file_out:
        pickle.dump(data, file_out)


def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)

        except EOFError:
            pass


def get_filenames(path):
    return next(walk(path), (None, None, []))[2]


def ftoMIDI(f):
    return 12 * np.log2(f / 440) + 69


PATHS = [
    "data/contours/expressive/extended/", "data/contours/expressive/urmp/"
]
SAMPLE_LEN = 1024
SAVE_PATH = "data/contours/expressive/dataset.pickle"

f0 = lo = np.array([])
for path in PATHS:
    print(f"==== Extracting from {path} directory ====")
    for file in get_filenames(path):
        print("Extracting from <- ", file)
        length = 0
        for c in read_from_pickle(path + file):
            length += len(c["f0"])
            f0 = np.concatenate((f0, c["f0"]))
            lo = np.concatenate((lo, c["lo"]))
        print(f"Extraction finished. {length//100}s added.")

print(f"Total length: {len(f0)//(100*60)}min{(len(f0)%(100*60))//100}s")
f0_chunks = [f0[i:i + SAMPLE_LEN] for i in range(0, len(f0), SAMPLE_LEN)]
lo_chunks = [lo[i:i + SAMPLE_LEN] for i in range(0, len(lo), SAMPLE_LEN)]

# shuffle
shuffle(f0_chunks)
shuffle(lo_chunks)

for fs, ls in zip(f0_chunks, lo_chunks):
    data = {"f0": ftoMIDI(fs), "lo": ls}
    export(data, SAVE_PATH)
