from tqdm import tqdm
from stockfish import Stockfish
import torch
from pprint import pprint
from alphaminustwo.dataset import get_val_loader, collate_fn_fen
from alphaminustwo.baselines import stockfish_eval


if __name__ == "__main__":
    data_path: str = "data/lichess_db_eval.jsonl"
    val_loader_fen = get_val_loader(
        data_path, bsz=8, val_size=250_000, num_workers=0, collate_fn=collate_fn_fen
    )
    baseline_sf_dict = {}
    for depth in [1]:
        stockfish = Stockfish(path="/opt/homebrew/bin/stockfish", depth=depth)
        loss = []
        for fen_list, y, z in tqdm(val_loader_fen):
            pred = torch.tensor([stockfish_eval(fen, stockfish) for fen in fen_list])
            loss.append(torch.nn.functional.mse_loss(pred, y))
        baseline_sf_dict[depth] = torch.tensor(loss).mean().item()
    print("stockfish")
    pprint(baseline_sf_dict)
