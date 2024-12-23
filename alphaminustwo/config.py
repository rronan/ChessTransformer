from dataclasses import dataclass
import os


@dataclass
class ModelCFG:
    block_size: int = 64
    square_dim: int = 18
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    weight_loss_move: float = 0.5


@dataclass
class TrainCFG:
    data_path: str = (
        os.environ["HOME"]
        + "/.cache/kagglehub/datasets/lichess/chess-evaluations/versions/3/dedups_evals.csv"
    )
    bsz: int = 512  # gpt:480
    val_size = 250_000
    val_interval: int = 2000
    compile: bool = True
    start_with_eval: bool = True
    log_interval: int = 100
    log_dir: str = "log"
    max_steps: int = 263_000  # gpt:600_000
    grad_clip: float = 1.0
    manual_seed = 1
    weight_decay: float = 0.1
    lr: float = 6e-4
    lr_start_factor: int = 0.1
    linear_warmup_iters: int = 2_000
    lr_end_factor: float = 0.1
    cosine_annealing_iters: int = 263_000
    beta1: float = 0.9
    beta2: float = 0.95
    watch_model: bool = False

    assert val_interval % log_interval == 0


model_cfg = ModelCFG()
train_cfg = TrainCFG()
