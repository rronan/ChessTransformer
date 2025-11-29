import logging
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import math
import chess

PIECES_CHAR = "PNBRQKpnbrqk"


def fen2tensor(s: str, extra_embedding: bool = True) -> torch.Tensor:
    # squares_embedding 64x13
    pieces_long = torch.tensor([ord(c) for c in PIECES_CHAR]).long().unsqueeze(0)
    pos, mov, castle, en_passant = s.split(" ")[:4]
    pos = pos.replace("/", "")
    pos_list = []
    for c in pos:
        if "1" <= c <= "8":
            pos_list.append(torch.zeros(int(c), 12))
        else:
            pos_list.append(pieces_long == ord(c))
    pos_tensor = torch.cat(pos_list, dim=0)
    assert pos_tensor.shape[0] == 64
    en_passant_tensor = torch.zeros(64, 1)
    if en_passant != "-":
        letter, number = en_passant
        index = 64 - (ord(letter) - ord("a")) * 8 - int(number)
        en_passant_tensor[index] = 1
    squares_embedding = torch.cat([pos_tensor, en_passant_tensor], dim=1)
    # extra_embedding 1x13
    mov_tensor = torch.zeros(1, 1) + int(mov == "w")
    castle_tensor = torch.zeros(1, 4)
    for k, v in enumerate("KQkq"):
        if v in castle:
            castle_tensor[:, k] = 1
    game_embedding = torch.cat([mov_tensor, castle_tensor], dim=1)
    if extra_embedding:
        e = torch.cat([game_embedding, torch.zeros(1, 8)], dim=1)
        res = torch.cat([e, squares_embedding], dim=0)
    else:
        e = game_embedding.expand(64, -1)
        res = torch.cat([squares_embedding, e], dim=1)
    return res


def tensor2fen(x: torch.Tensor) -> str:
    raise NotImplementedError


def tensor2str(x: torch.Tensor):
    board = x[1:, :12].reshape(8, 8, 12)
    res = ""
    for row in board:
        res += "|"
        for square in row:
            piece = " "
            for piece_index, piece_value in enumerate(square):
                if piece_value == 1:
                    piece = PIECES_CHAR[piece_index]
                    break
            res += piece
        res += "|\n"
    return res


def invert_color(x: torch.Tensor, y: torch.Tensor):
    y = torch.zeros_like(x)
    y[0, :4] = x[0, 4::-1]
    y[0, 5] = 1 - y[0, 5]
    y[1:] = y[-1:0:-1]
    y[1:, :12] = y[1:, 12::-1]
    return y


def uci2index(s: str):
    squares = []
    for k in [0, 1]:
        letter, number = s[2 * k : 2 * k + 2]
        index = (ord(letter) - ord("a")) * 8 + int(number) - 1
        squares.append(index)
    res = squares[0] * 64 + squares[1]
    return torch.tensor(res)


def index2uci(index: int):
    res = ""
    for square in [index // 64, index % 64]:
        i, j = square // 8, square % 8
        res += list("abcdefgh")[i]
        res += str(j + 1)
    return res


def process_evaluation(y: dict) -> float:
    if y["mate"] is not None:
        return float(y["mate"] > 0)
    return 1 / (1 + math.exp(-0.00368208 * y["cp"]))


class LineAugmentedDataset(IterableDataset):
    def __init__(
        self,
        base_dataset: Dataset,
        n_max: int | None,
        min_depth: int | None,
        extra_embedding: bool,
    ):
        self.base_dataset = base_dataset
        self.n_max = max(n_max, 1)
        self.min_depth = min_depth
        self.extra_embedding = extra_embedding

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = info.id
            num_workers = info.num_workers
        for idx, item in enumerate(self.base_dataset):
            # Skip items that don't belong to this worker
            if idx % num_workers != worker_id:
                continue
            first_eval = item["evals"][0]  # deepest eval
            depth = first_eval["depth"]
            pv = first_eval["pvs"][0]  # best line
            y = torch.tensor(process_evaluation(pv)).float()
            try:
                board = chess.Board(item["fen"])
            except Exception as e:
                logging.debug(f"Error parsing fen: {item['fen']}: {e}")
                continue
            for i, move in enumerate(pv["line"].split(" ")[: self.n_max]):
                if self.min_depth is not None and (depth - i) < self.min_depth:
                    break
                x = fen2tensor(board.fen(), self.extra_embedding)
                z = uci2index(move)
                try:
                    board.push_uci(move)
                except chess.IllegalMoveError as e:
                    logging.debug(f"Illegal move: {move} in {board.fen()}")
                    continue
                yield x.float(), y, z.long()


def get_train_loader_line_augmented(
    data_path, bsz, n_max, min_depth, extra_embedding, num_workers, shuffle
):
    assert not (num_workers > 0 and shuffle)
    dataset_train = load_dataset(
        "json", data_files=data_path, split="train", streaming=num_workers == 0
    )
    if shuffle:
        dataset_train = dataset_train.shuffle(buffer_size=10000)
    train_loader = DataLoader(
        LineAugmentedDataset(
            dataset_train,
            n_max=n_max,
            min_depth=min_depth,
            extra_embedding=extra_embedding,
        ),
        batch_size=bsz,
    )
    return iter(train_loader)


class DataStats:
    mean: float = 0.5421502590179443
    std: float = 0.24764062464237213
    var: float = 0.06132587897326425
    stockfish_1: float = 0.6568744778633118
    count_pieces: float = 0.29602745175361633
    bce: float = 0.6931473016738892


# LEGACY CODE - TESTING


def process_best_move(line):
    best_move = line.split(" ")[0]
    return uci2index(best_move)


def collate_fn(x_list):
    fens, evaluations, lines = [], [], []
    for item in x_list:
        fen, pv = item["fen"], item["evals"][0]["pvs"][0]
        fens.append(fen)
        evaluations.append(pv)
        lines.append(pv["line"])
    x = torch.stack([fen2tensor(fen) for fen in fens])
    y = torch.tensor([process_evaluation(eval_pv) for eval_pv in evaluations])
    z = torch.stack([process_best_move(line) for line in lines])
    return x.float(), y.float(), z.long()


def get_train_loader(data_path, bsz, num_workers, shuffle, collate_fn=collate_fn):
    dataset_train = load_dataset("json", data_files=data_path, split="train")
    train_loader = DataLoader(
        dataset_train,
        batch_size=bsz,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return train_loader
