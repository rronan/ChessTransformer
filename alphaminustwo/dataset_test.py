import pytest
import math
from tqdm import tqdm
import sys
from itertools import islice
from alphaminustwo.dataset import (
    fen2tensor,
    tensor2str,
    process_evaluation,
    process_best_move,
    get_train_loader_line_augmented,
    get_train_loader,
)
import numpy as np
import torch
import time

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


def _test_loader(loader, limit=None) -> int:
    c = 0
    for x, y, z in tqdm(loader):
        if limit is not None and c >= limit:
            break
        c += 1
        bsz = x.shape[0]
        assert torch.all((x[:, 1:, :-1].sum(-1) >= 0) & (x[:, 1:, :-1].sum(-1) <= 1))
        assert x.shape == (bsz, 65, 13)
        assert y.shape == (bsz,)
        assert z.shape == (bsz,)
    return c


@pytest.mark.parametrize(
    "bsz,n_max,min_depth,num_workers,shuffle,limit,extra_embedding",
    [
        (1, 1, None, 0, False, 10, True),
        (128, 10, 20, 4, True, 100, True),
        (128, None, 20, 1, False, 100, True),
        (128, 3, 20, 6, False, None, True),
        (128, 3, 20, 6, False, None, False),
    ],
)
def test_line_augmented_train_loader(
    bsz, n_max, min_depth, num_workers, shuffle, limit, extra_embedding
):
    train_loader = get_train_loader_line_augmented(
        DATA_PATH,
        bsz=bsz,
        val_size=1,
        n_max=n_max,
        min_depth=min_depth,
        num_workers=num_workers,
        extra_embedding=extra_embedding,
        shuffle=shuffle,
    )
    count = _test_loader(train_loader, limit)
    print(
        f"{bsz=}, {n_max=}, {min_depth=}, {num_workers=}, {shuffle=}, {limit=}, {count=}"
    )


# def test_line_augmented_train_loader_wrt_normal_train_loader():
#     """
#     This test allowed us to find illegal fens in the dataset, e.g.:
#     `r2qk3/2p1p1pr/1p1pPp1p/p1nPb1N1/3B4/2N5/PPP2PPP/R2Q1RK1 b q - 0 1`
#     """
#     train_loader = get_train_loader(DATA_PATH, bsz=1, val_size=1, shuffle=False)
#     line_augmented_train_loader = get_train_loader_line_augmented(
#         DATA_PATH,
#         bsz=1,
#         val_size=1,
#         n_max=1,
#         min_depth=None,
#         num_workers=0,
#         shuffle=False,
#     )
#     MAX_ITER = 1000
#     c = 0
#     for (x, y, z), (x_line_augmented, y_line_augmented, z_line_augmented) in tqdm(
#         zip(train_loader, line_augmented_train_loader), miniters=1
#     ):
#         c += 1
#         if c > MAX_ITER:
#             break
#         assert x.shape == x_line_augmented.shape
#         assert y.shape == y_line_augmented.shape
#         assert z.shape == z_line_augmented.shape
#         assert torch.allclose(x, x_line_augmented)
#         assert torch.allclose(y, y_line_augmented)
#         assert torch.allclose(z, z_line_augmented)


def speedtest_train_loader(data_path, bsz, num_workers, shuffle, max_iter):
    print(f"train_loader speed test: {bsz=}, {num_workers=}, {shuffle=}, {max_iter=}")
    train_loader = get_train_loader(
        data_path=data_path, bsz=bsz, num_workers=num_workers, shuffle=shuffle
    )
    start_time = time.time()
    for _ in tqdm(islice(train_loader, max_iter), total=max_iter):
        pass
    end_time = time.time()
    print(f"Success: {end_time - start_time} seconds")


def speedtest_train_loader_line_augmented(
    data_path, bsz, n_max, min_depth, shuffle, num_workers, max_iter
):
    print(
        f"train_loader_line_augmented speed test: {bsz=}, {num_workers=}, {n_max=}, {min_depth=}, {max_iter=}, {shuffle=}"
    )
    train_loader = get_train_loader_line_augmented(
        data_path=data_path,
        bsz=bsz,
        n_max=n_max,
        min_depth=min_depth,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    start_time = time.time()
    for _ in tqdm(islice(train_loader, max_iter), total=max_iter):
        pass
    end_time = time.time()
    print(f"Success: {end_time - start_time} seconds")


if __name__ == "__main__":
    num_workers = 8
    shuffle = False
    speedtest_train_loader(
        data_path="data/lichess_db_eval.jsonl",
        bsz=480,
        num_workers=num_workers,
        shuffle=shuffle,
        max_iter=100,
    )
    speedtest_train_loader_line_augmented(
        data_path="data/lichess_db_eval.jsonl",
        bsz=480,
        n_max=1,
        min_depth=None,
        shuffle=shuffle,
        num_workers=num_workers,
        max_iter=100,
    )
