from dataclasses import dataclass


@dataclass
class GPT124M:
    block_size: int = 65
    square_dim: int = 13
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    weight_loss_move: float = 1
    weight_loss_eval: float = 4 / 0.6931473016738892


@dataclass
class TrainCFG:
    data_path: str = "data/lichess_db_eval.jsonl"
    bsz: int = 360  # gpt:480
    val_size = 250_000
    val_interval: int = 2000
    compile: bool = True
    start_with_eval: bool = True
    log_interval: int = 100
    log_dir: str = "log"
    start_steps: int = 0
    max_steps: int = 600_000  # gpt:600_000
    grad_clip: float = 1.0
    manual_seed = 1
    weight_decay: float = 0.1
    lr: float = 6e-4
    lr_scheduler: str = "gpt2"
    lr_start_factor: int = 0.1
    linear_warmup_iters: int = 2_000
    lr_end_factor: float = 0.1
    cosine_annealing_iters: int = 600_000
    beta1: float = 0.9
    beta2: float = 0.95
    watch_model: bool = False
    wandb_resume_from: str = None  # "dkm3uzyd?_step=29"

    assert val_interval % log_interval == 0
