import chess
from argparse import ArgumentParser

import torch

from alphaminustwo.dataset import tensor2move, fen2tensor
from alphaminustwo.model import GPT
import alphaminustwo.config

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")  # on RTF4090, 40% speedup


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--chkp")
    parser.add_argument("--fen", nargs="+")
    parser.add_argument("--prompt", action="store_true")
    args = parser.parse_args()
    return args


args = parse_args()
model = GPT(alphaminustwo.model_cfg).to(device)
import sys
sys.modules["config"] = alphaminustwo.config
chkp = torch.load(args.chkp, weights_only=False, map_location=torch.device(device))
model.load_state_dict(chkp["model"])
model.eval()


def run_and_print(model, fen_list):
    x_list = [fen2tensor(s) for s in fen_list]
    x = torch.stack(x_list).to(device)
    with torch.no_grad():
        evals, moves, *_ = model(x, None)
        for fen, eval, move in zip(fen_list, evals, moves):
            board = chess.Board(fen)
            legal_moves = [x.uci() for x in board.legal_moves]
            print(legal_moves)
            print("https://lichess.org/analysis/fromPosition/" + fen.replace(" ", "_"))
            print(f"{eval * 60:.3f} - {tensor2move(move, legal_moves=legal_moves)}")


if args.fen is not None:
    run_and_print(model, args.fen)

if args.prompt:
    while True:
        s = input("Fen: ")
        run_and_print(model, [s])
