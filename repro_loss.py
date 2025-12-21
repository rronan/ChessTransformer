import torch
from alphaminustwo.model import GPT
from alphaminustwo import config


def test_loss_decrease():
    # Setup model and optimizer
    cfg = config.GPT124M()
    # User says "even if loss_move=0", likely meaning weight
    cfg.weight_loss_move = 0.0

    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)  # Default LR

    # Create synthetic data
    bsz = 8
    # x shape: [B, 64, 18] if extra_embedding=False
    x = torch.randn(bsz, 64, 18)

    # Make y depend on x in a simple linear way to ensure it's learnable
    # e.g. sum of first few elements
    target_logits = x.view(bsz, -1)[:, :10].sum(dim=1)
    y = torch.sigmoid(target_logits)

    z = torch.randint(0, 64 * 64, (bsz,))

    print("Starting training loop...")
    initial_loss = None

    for i in range(50):
        optimizer.zero_grad()
        _, _, loss_score, loss_move, loss = model(x, y, z)

        if i == 0:
            initial_loss = loss_score.item()
            print(f"Iter {i}: loss_score={initial_loss}")

        if i % 10 == 0:
            print(f"Iter {i}: loss_score={loss_score.item()}")

        loss.backward()
        optimizer.step()

    final_loss = loss_score.item()
    print(f"Final loss_score: {final_loss}")

    if final_loss < initial_loss:
        print("SUCCESS: loss_score decreased")
    else:
        print("FAILURE: loss_score did not decrease")


if __name__ == "__main__":
    test_loss_decrease()
