from tqdm import tqdm
from stockfish import Stockfish
import os
import torch
from pprint import pprint
from alphaminustwo.dataset import get_val_loader, collate_fn_fen
from alphaminustwo.baselines import count_pieces, stockfish_eval



if __name__ == "__main__":
    data_path: str = (
        os.environ["HOME"]
        + "/.cache/kagglehub/datasets/lichess/chess-evaluations/versions/3/dedups_evals.csv"
    )
    val_loader = get_val_loader(data_path, bsz=8, val_size=250_000, num_workers=0)
    y_list = []
    baseline_pc_list = []
    for x, y, _ in tqdm(val_loader):
        y_list.append(y)
        baseline_pc_list.append(torch.nn.functional.mse_loss(count_pieces(x), y))
    y = torch.cat(y_list, dim=0)
    std = y.std().item()
    print("mean:", y.mean().item(), "std:", std, "var:", std**2)
    val_loader_fen = get_val_loader(
        data_path, bsz=8, val_size=250_000, num_workers=0, collate_fn=collate_fn_fen
    )
    baseline_sf_dict = {}
    for depth in [1]:
        stockfish = Stockfish(path="/usr/games/stockfish", depth=depth)
        loss = []
        for fen_list, y, z in tqdm(val_loader_fen):
            pred = torch.tensor([stockfish_eval(fen, stockfish) for fen in fen_list])
            loss.append(torch.nn.functional.mse_loss(pred, y))
        baseline_sf_dict[depth] = torch.tensor(loss).mean().item()
    print("stockfish")
    pprint(baseline_sf_dict)
