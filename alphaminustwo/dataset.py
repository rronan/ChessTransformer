import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import math

PIECES_CHAR = "PNBRQKpnbrqk"


def fen2tensor(s: str) -> torch.Tensor:
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
    extra_embedding = torch.cat([mov_tensor, castle_tensor, torch.zeros(1, 8)], dim=1)
    res = torch.cat([extra_embedding, squares_embedding], dim=0)
    return res


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


def tensor2fen(x: torch.Tensor) -> str:
    raise NotImplementedError


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



def process_evaluation(y):
    if y["mate"] is not None:
        return y["mate"] > 0
    return 1 / (1 + math.exp(-0.00368208 * y["cp"]))


def move2tensor(s: str):
    squares = []
    for k in [0, 1]:
        letter, number = s[2 * k : 2 * k + 2]
        index = (ord(letter) - ord("a")) * 8 + int(number) - 1
        squares.append(index)
    res = squares[0] * 64 + squares[1]
    return torch.tensor(res)

def extract_evaluation_data(item):
    fen = item["fen"]
    first_eval = item["evals"][0]
    pv = first_eval["pvs"][0]
    return fen, pv


def process_best_move(line):
    best_move = line.split(" ")[0]
    return uci2index(best_move)


def collate_fn(x_list):
    fens, evaluations, lines = [], [], []
    for item in x_list:
        fen, pv = extract_evaluation_data(item)
        fens.append(fen)
        evaluations.append(pv)
        lines.append(pv["line"])
    x = torch.stack([fen2tensor(fen) for fen in fens])
    y = torch.tensor([process_evaluation(eval_pv) for eval_pv in evaluations])
    z = torch.stack([process_best_move(line) for line in lines])
    return x.float(), y.float(), z.long()


def collate_fn_fen(x_list):
    fens, evaluations, lines = [], [], []
    for item in x_list:
        fen, pv = extract_evaluation_data(item)
        fens.append(fen)
        evaluations.append(pv)
        lines.append(pv["line"])

    y = torch.tensor([process_evaluation(eval_pv) for eval_pv in evaluations])
    z = torch.stack([process_best_move(line) for line in lines])
    return fens, y.float(), z.long()


def get_train_loader(data_path, bsz, val_size, num_workers=8):
    dataset_train = load_dataset(
        "json", data_files=data_path, split=f"train[:-{val_size}]"
    )
    train_loader = iter(
        DataLoader(
            dataset_train,
            batch_size=bsz,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )
    )
    return train_loader


def get_val_loader(data_path, bsz, val_size, num_workers=8, collate_fn=collate_fn):
    dataset_val = load_dataset(
        "json", data_files=data_path, split=f"train[-{val_size}:]"
    )
    val_loader = DataLoader(
        dataset_val, batch_size=bsz, num_workers=num_workers, collate_fn=collate_fn
    )
    return val_loader


class DataStats:
    var = 1.2523940917372443
    stockfish_1: 0.6568744778633118
