from dataclasses import dataclass


@dataclass
class CommonConfig:
    extra_embedding: bool = False


@dataclass
class GPT124M:
    square_dim: int = 13 if CommonConfig.extra_embedding else 18
    extra_embedding: bool = CommonConfig.extra_embedding
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    weight_loss_move: float = 1
    weight_loss_score: float = 4


@dataclass
class GPT345M:
    square_dim: int = 13 if CommonConfig.extra_embedding else 18
    extra_embedding: bool = CommonConfig.extra_embedding
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    bias: bool = False
    weight_loss_move: float = 1
    weight_loss_score: float = 3


# Alias for compatibility
GPT124M = ModelCFG


@dataclass
class TrainCFG:
    data_path: str = "data/lichess_db_eval.jsonl"
    puzzle_path: str = "data/lichess_db_puzzle.csv"
    extra_embedding: bool = CommonConfig.extra_embedding
    shuffle: bool = True
    num_workers: int = 0
    initial_puzzle_elo: float = 170
    n_puzzles = 1000
    checkpoint_interval: int = 5000
    bsz: int = 512  # gpt:480
    n_max: int = 0
    min_depth: int = 21
    accumulate_grad_steps: int = 1  # gpt:1
    compile: bool = True
    log_interval: int = 100
    log_dir: str = "log"
    max_steps: int = 600_000  # gpt:600_000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    lr: float = 6e-4
    lr_scheduler: str = "gpt2"
    lr_start_factor: int = 0.1
    linear_warmup_iters: int = 2_000
    lr_end_factor: float = 0.1
    cosine_annealing_iters: int = 600_000
    beta1: float = 0.9
    beta2: float = 0.95

    assert checkpoint_interval % log_interval == 0
