import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def fen2tensor(s: str) -> torch.Tensor:
    pieces_char = "PNBRQKpnbrqk"
    pieces_long = torch.tensor([ord(c) for c in pieces_char]).long().unsqueeze(0)
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
    mov_tensor = torch.zeros(64, 1) + int(mov == "w")
    castle_tensor = torch.zeros(64, 4)
    for k, v in enumerate("KQkq"):
        if v in castle:
            castle_tensor[:, k] = 1
    en_passant_tensor = torch.zeros(64, 1)
    if en_passant != "-":
        letter, number = en_passant
        index = 64 - (ord(letter) - ord("a")) * 8 - int(number)
        en_passant_tensor[index] = 1
    res = torch.cat([pos_tensor, mov_tensor, castle_tensor, en_passant_tensor], dim=1)
    return res


def tensor2fen(x: torch.Tensor) -> str:
    raise NotImplementedError


def invert_color(x: torch.Tensor, y: torch.Tensor):
    raise NotImplementedError


def process_mate(m):
    sign = m / abs(m)
    scale = abs(m) - 1
    return sign * max(30, 60 - scale * 2)


def process_evaluation(y):
    if y["cp"] is not None:
        res = y["cp"] / 100
    elif y["mate"] is not None:
        res = process_mate(y["mate"])
    else:
        raise AttributeError
    scale = 4
    return scale * list(sorted([-60, res, 60]))[1] / 60.0

def move2tensor(s: str):
    squares = []
    for k in [0, 1]:
        letter, number = s[2 * k : 2 * k + 2]
        index = 64 - (ord(letter) - ord("a")) * 8 - int(number)
        squares.append(index)
    res = squares[0] * 64 + squares[1]
    return torch.tensor(res)


def process_best_move(line):
    best_move = line.split(" ")[0]
    return move2tensor(best_move)



def tensor2move(x: torch.Tensor, legal_moves=None):
    res = ""
    if legal_moves is not None:
        mask = torch.zeros(64**2)
        for move in legal_moves:
            mask[move2tensor(move)] = 1
        x.masked_fill_(mask==0, float('-inf'))
    index = torch.multinomial(x.exp(), 1)[0].item()
    print("Distribution:", [f"{xx.item():.4f}" for xx in x.exp() if xx > 0])
    square_list = [index // 64, index % 64]
    for square in square_list:
        i, j = square // 8, square % 8
        res += list("hgfedcba")[i]
        res += str(8 - j)
    return res


def collate_fn(x_list):
    x = torch.stack([fen2tensor(x["fen"]) for x in x_list])
    y = torch.tensor([process_evaluation(x) for x in x_list])
    z = torch.stack([process_best_move(x["line"]) for x in x_list])
    return x.float(), y.float(), z.long()


def collate_fn_fen(x_list):
    fen_list = [x["fen"] for x in x_list]
    y = torch.tensor([process_evaluation(x) for x in x_list])
    z = torch.stack([process_best_move(x["line"]) for x in x_list])
    return fen_list, y.float(), z.long()


def get_train_loader(data_path, bsz, val_size, num_workers=8):
    dataset_train = load_dataset(
        "csv", data_files=data_path, split=f"train[:-{val_size}]"
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
        "csv", data_files=data_path, split=f"train[-{val_size}:]"
    )
    val_loader = DataLoader(
        dataset_val, batch_size=bsz, num_workers=num_workers, collate_fn=collate_fn
    )
    return val_loader


class DataStats:
    var = 1.2523940917372443
    std: 1.1191041469573975
    stockfish_1: 1.0465227365493774
