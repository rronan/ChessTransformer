import os
import sys
import chess
from chess.engine import PlayResult
import torch
from .model import GPT
from . import config

# https://github.com/lichess-bot-devs/lichess-bot
from lib.engine_wrapper import MinimalEngine
from lib.types import HOMEMADE_ARGS_TYPE

"""
Usage:

- Import AlphaMinusTwo in lichess-bot/homemade.py
- Configure as per: https://github.com/lichess-bot-devs/lichess-bot/wiki/Create-a-homemade-engine
- Set env var ALPHAMINUSTWO_CHKP=<PATH> and run the bot
"""

CHKP = os.environ["ALPHAMINUSTWO_CHKP"]

class AlphaMinusTwo(MinimalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cfg = config.ModelCFG()
        self.model = GPT(model_cfg).to(self.device)
        # temp fix
        sys.modules["config"] = config
        chkp = torch.load(
            CHKP,
            weights_only=False,
            map_location=torch.device(self.device),
        )
        self.model.load_state_dict(chkp["model"])
        self.model.eval()

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        move_list, eval_list = self.model.generate_from_board([board], legal_move=True)
        print(eval_list[0])
        return PlayResult(move_list[0], None)