import chess
from argparse import ArgumentParser

import torch

from alphaminustwo.model import GPT
from alphaminustwo.config import ModelCFG

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")  # on RTF4090, 40% speedup

model_cfg = ModelCFG()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--chkp")
    parser.add_argument("--fen", nargs="+")
    parser.add_argument("--prompt", action="store_true")
    args = parser.parse_args()
    return args


args = parse_args()
model = GPT(model_cfg).to(device)

chkp = torch.load(args.chkp, weights_only=False, map_location=torch.device(device))
model.load_state_dict(chkp["model"])
model.eval()


def run_and_print(model, fen_list):
    board_list = [chess.Board(fen=fen) for fen in fen_list]
    move_list, eval_list = model.generate_from_board(board_list)
    for fen, move, eval in zip(fen_list, move_list, eval_list):
        print("https://lichess.org/analysis/fromPosition/" + fen.replace(" ", "_"))
        print(f"{eval * 60:.3f} - {move.uci()}")


if args.fen is not None:
    run_and_print(model, args.fen)

if args.prompt:
    while True:
        s = input("Fen: ")
        run_and_print(model, [s])
