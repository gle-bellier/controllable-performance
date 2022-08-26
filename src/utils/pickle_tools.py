import pickle
import numpy as np
from typing import Tuple


def read_from_pickle(path: str) -> dict:
    """Read elements from a pickle file.

    Args:
        path (str): path to the pickle file.


    Yields:
        Iterator[dict]: element from the pickle
        file.
    """

    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)

        except EOFError:
            pass
