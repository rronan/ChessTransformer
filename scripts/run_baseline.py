from tqdm import tqdm
from stockfish import Stockfish
import torch
from datasets import load_dataset
from pprint import pprint
from alphaminustwo.dataset import process_evaluation, fen2tensor
from alphaminustwo.baselines import stockfish_score, count_pieces


if __name__ == "__main__":
    data_path: str = "data/lichess_db_eval.jsonl"
    depth = 1
    extra_embedding = True
    val_dataset = load_dataset("json", data_files=data_path, split="train[:100000]")
    stockfish = Stockfish(path="/opt/homebrew/bin/stockfish", depth=depth)
    loss_pc_mse, loss_sf_mse, loss_pc_bce, loss_sf_bce, y_list = [], [], [], [], []
    for item in tqdm(val_dataset):
        y = torch.tensor(process_evaluation(item["evals"][0]["pvs"][0]))
        y_list.append(y)
        x = fen2tensor(item["fen"], extra_embedding=extra_embedding)
        count_pieces_y = torch.tensor(
            process_evaluation({"cp": count_pieces(x), "mate": None})
        )
        stockfish_y = torch.tensor(stockfish_score(item["fen"], stockfish=stockfish))
        loss_pc_bce.append(torch.nn.functional.binary_cross_entropy(count_pieces_y, y))
        loss_sf_bce.append(torch.nn.functional.binary_cross_entropy(stockfish_y, y))
        loss_pc_mse.append(torch.nn.functional.mse_loss(count_pieces_y, y))
        loss_sf_mse.append(torch.nn.functional.mse_loss(stockfish_y, y))
    pprint(
        {
            "y_mean": torch.stack(y_list, dim=0).mean().item(),
            "y_std": torch.stack(y_list, dim=0).std().item(),
            "count_pieces_loss_mse": torch.stack(loss_pc_mse, dim=0).mean().item(),
            "count_pieces_loss_bce": torch.stack(loss_pc_bce, dim=0).mean().item(),
            "stockfish_loss_mse": torch.stack(loss_sf_mse, dim=0).mean().item(),
            "stockfish_loss_bce": torch.stack(loss_sf_bce, dim=0).mean().item(),
        }
    )
