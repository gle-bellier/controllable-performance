import pickle
from typing import List, Tuple
from os import walk
import numpy as np


def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)

        except EOFError:
            pass


def export(f0: np.ndarray, lo: np.ndarray, path: str) -> None:
    data = {"f0": f0, "lo": lo}
    with open(path, "ab+") as file_out:
        pickle.dump(data, file_out)


def split_indexes(N, prop) -> List[Tuple[int]]:

    train_start, train_end = 0, int(N * prop[0])
    val_start, val_end = train_end, train_end + int(N * prop[1])
    eval_start, eval_end = val_end, val_end + int(N * prop[2])

    return [(train_start, train_end), (val_start, val_end),
            (eval_start, eval_end)]


PATH = "data/contours/expressive/dataset.pickle"
prop = [0.8, 0.1, 0.1]

f0 = lo = np.array([])

for c in read_from_pickle(PATH):
    f0 = np.concatenate((f0, c["f0"]))
    lo = np.concatenate((lo, c["lo"]))

N = len(lo)
train_idx, val_idx, eval_idx = split_indexes(N, prop)

train_f0, train_lo = f0[slice(*train_idx)], lo[slice(*train_idx)]
val_f0, val_lo = f0[slice(*val_idx)], lo[slice(*val_idx)]
eval_f0, eval_lo = f0[slice(*eval_idx)], lo[slice(*eval_idx)]

export(train_f0, train_lo, "data/contours/expressive/train.pickle")
export(val_f0, val_lo, "data/contours/expressive/val.pickle")
export(eval_f0, eval_lo, "data/contours/expressive/test.pickle")