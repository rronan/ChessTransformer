import torch
from alphaminustwo.model import GPT
from alphaminustwo import config
from alphaminustwo.dataset import get_train_loader_line_augmented
from tqdm import tqdm


def overfit_single_batch():
    # Setup
    train_cfg = config.TrainCFG()
    # Force extra_embedding to match model config if needed, but default is False for both
    cfg = config.GPT124M()

    print(f"Model config extra_embedding: {cfg.extra_embedding}")

    model = GPT(cfg)
    model.train()

    # Use CPU for determinism/simplicity if CUDA not avail, but check CUDA if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # Get one batch
    loader = get_train_loader_line_augmented(
        data_path=train_cfg.data_path,
        bsz=32,  # Small batch for overfitting
        n_max=train_cfg.n_max,
        min_depth=train_cfg.min_depth,
        num_workers=0,
        extra_embedding=cfg.extra_embedding,
        shuffle=True,
    )

    x, y, z = next(loader)
    x, y, z = x.to(device), y.to(device), z.to(device)

    print(f"Input x shape: {x.shape}")
    print(f"Target y shape: {y.shape}")
    print(f"Target y mean: {y.mean().item():.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nStarting overfitting loop...")
    for i in tqdm(range(100)):
        optimizer.zero_grad()
        y_score, _, loss_score, _, loss = model(x, y, z)

        if i % 10 == 0:
            score_grad_norm = 0.0
            for param in model.score_head.parameters():
                if param.grad is not None:
                    score_grad_norm += param.grad.norm().item()

            probs = torch.sigmoid(y_score)
            print(
                f"Iter {i:03d} | Loss: {loss.item():.4f} | Loss Score: {loss_score.item():.4f} | Score Grad: {score_grad_norm:.4f} | Pred Mean: {probs.mean().item():.4f}"
            )

        loss.backward()
        optimizer.step()

    print("\nFinal predictions vs targets:")
    probs = torch.sigmoid(y_score).detach().cpu()
    targets = y.cpu()
    for j in range(5):
        print(f"Target: {targets[j]:.4f} | Pred: {probs[j]:.4f}")


if __name__ == "__main__":
    overfit_single_batch()
