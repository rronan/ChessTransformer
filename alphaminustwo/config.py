from dataclasses import dataclass


@dataclass
class GPT124M:
    block_size: int = 65
    square_dim: int = 13
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    extra_embedding: bool = True
    weight_loss_move: float = 3
    weight_loss_eval: float = 0
    # weight_loss_move: float = 1
    # weight_loss_eval: float = 4 / 0.6931473016738892


@dataclass
class GPT345M:
    block_size: int = 65
    square_dim: int = 13
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    bias: bool = False
    weight_loss_move: float = 1
    weight_loss_eval: float = 4 / 0.6931473016738892


@dataclass
class TrainCFG:
    data_path: str = "data/lichess_db_eval.jsonl"
    puzzle_path: str = "data/lichess_db_puzzle.csv"
    num_workers: int = 0
    initial_puzzle_elo: float = 170
    n_puzzles = 1000
    checkpoint_interval: int = 5000
    bsz: int = 90  # gpt:480
    n_max: int = 3
    min_depth: int = 21
    accumulate_grad_steps: int = 4  # gpt:1
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
