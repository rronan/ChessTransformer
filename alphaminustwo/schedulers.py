from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler


def get_scheduler(optimizer, train_cfg):
    if train_cfg.lr_scheduler == "dummy":
        return LinearLR(optimizer, start_factor=1)
    if train_cfg.lr_scheduler == "gpt2":
        return ChainedScheduler(
            [
                LinearLR(
                    optimizer,
                    start_factor=train_cfg.lr_start_factor,
                    total_iters=train_cfg.linear_warmup_iters,
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=train_cfg.cosine_annealing_iters,
                    eta_min=train_cfg.lr * train_cfg.lr_end_factor,
                ),
            ]
        )
