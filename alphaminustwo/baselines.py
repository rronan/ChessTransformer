import torch
from stockfish import Stockfish
from alphaminustwo.dataset import process_evaluation


def count_pieces(x):
    values = [1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, -0]
    x = x[:, :, :12]
    weighted_x = x * torch.tensor(values).unsqueeze(0).unsqueeze(0)
    res = weighted_x.sum(-1).sum(-1) / 60.0
    return res


def stockfish_eval(fen, stockfish=None):
    if stockfish is None:
        Stockfish(path="/usr/games/stockfish")
    stockfish.set_fen_position(fen)
    y = stockfish.get_evaluation()
    return process_evaluation(y)
