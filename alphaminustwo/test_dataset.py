import math
import torch
from alphaminustwo.dataset import get_train_loader, get_val_loader
from alphaminustwo.dataset import fen2tensor
from alphaminustwo.dataset import tensor2str
from alphaminustwo.dataset import process_evaluation
from alphaminustwo.dataset import process_best_move
import numpy as np

# TODO invert_colors two times and check it is equal
DATA_PATH = "data/lichess_db_eval.jsonl"


def test_fen():
    fen = "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -"
    expected_str = """\
|       r|
| p   k  |
|p bPR   |
|     p  |
|  B  P p|
|        |
|PP    P |
|   K    |
"""
    x = fen2tensor(fen)
    s = tensor2str(x)
    assert s == expected_str
    assert x.shape == (65, 13)
    assert x.unique()[0] == 0 and x.unique()[1] == 1


def test_process_evaluation():
    y = {"cp": 100, "mate": None}
    expected_y = 1 / (1 + math.exp(-0.00368208 * 100))
    assert np.isclose(process_evaluation(y), expected_y)
    y = {"cp": -100, "mate": None}
    assert np.isclose(process_evaluation(y), 1 - expected_y)


def test_process_best_move():
    line = "a1b1"
    res = 0 * 64 + 8
    processed_line = process_best_move(line)
    assert processed_line == res
    line = "a1a2"
    expected_line = 0 * 64 + 1
    processed_line = process_best_move(line)
    assert processed_line == expected_line
    line = "b1a1"
    expected_line = 8 * 64 + 0
    processed_line = process_best_move(line)
    assert processed_line == expected_line


def _test_loader(loader):
    for x, y, z in loader:
        assert x.shape == (128, 65, 13)
        assert y.shape == (128,)
        assert z.shape == (128,)
        break


def test_train_loader():
    train_loader = get_train_loader(DATA_PATH, 128, 100000)
    _test_loader(train_loader)


def test_val_loader():
    val_loader = get_val_loader(DATA_PATH, 128, 100000)
    _test_loader(val_loader)


if __name__ == "__main__":
    train_loader = get_train_loader(DATA_PATH, 1, 10)
    for x, y, z in train_loader:
        print(x.shape, y.shape, z.shape)
        print(x)
        print(y)
        print(z)
        break
