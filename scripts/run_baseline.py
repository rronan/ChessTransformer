from tqdm import tqdm
from stockfish import Stockfish
import torch
from datasets import load_dataset
from pprint import pprint
from alphaminustwo.dataset import process_evaluation, fen2tensor
from alphaminustwo.baselines import stockfish_eval, count_pieces


if __name__ == "__main__":
    data_path: str = "data/lichess_db_eval.jsonl"
    val_dataset = load_dataset("json", data_files=data_path, split="train[-250000:]")
    depth = 1
    stockfish = Stockfish(path="/opt/homebrew/bin/stockfish", depth=depth)
    loss_pc, loss_sf = [], []
    for item in tqdm(val_dataset):
        y = process_evaluation(item["evals"][0]["pvs"][0])
        x = fen2tensor(item["fen"])
        count_pieces_y = process_evaluation(count_pieces(x))
        loss_pc.append(torch.nn.functional.binary_cross_entropy(count_pieces_y, y))
        stockfish_y = process_evaluation(
            stockfish_eval(item["fen"], stockfish=stockfish)
        )
        loss_sf.append(torch.nn.functional.binary_cross_entropy(stockfish_y, y))
    print("stockfish")
    pprint(torch.tensor(loss_sf).mean().item(), torch.tensor(loss_sf).std().item())
    print("count_pieces")
    pprint(torch.tensor(loss_pc).mean().item(), torch.tensor(loss_pc).std().item())
