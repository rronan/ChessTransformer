from tqdm import tqdm
import torch
from alphaminustwo.dataset import get_val_loader
from alphaminustwo.baselines import count_pieces


if __name__ == "__main__":
    data_path: str = "data/lichess_db_eval.jsonl"
    val_loader = get_val_loader(data_path, bsz=8, val_size=250_000, num_workers=0)
    y_list = []
    baseline_pc_list = []
    baseline_bce_list = []
    for x, y, _ in tqdm(val_loader):
        y_list.append(y)
        baseline_pc_list.append(torch.nn.functional.mse_loss(count_pieces(x), y))
        baseline_bce_list.append(
            torch.nn.functional.binary_cross_entropy_with_logits(torch.zeros_like(y), y)
        )
    y = torch.cat(y_list, dim=0)
    std = y.std().item()
    print("mean:", y.mean().item(), "std:", std, "var:", std**2)
    print("baseline_pc_list:", torch.tensor(baseline_pc_list).mean().item())
    print("baseline_bce_list:", torch.tensor(baseline_bce_list).mean().item())
