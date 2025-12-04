import torch
from alphaminustwo.dataset import get_train_loader_line_augmented
from alphaminustwo import config
from tqdm import tqdm

def inspect_data():
    train_cfg = config.TrainCFG()
    # Set num_workers to 0 to avoid multiprocessing issues in script
    train_loader = get_train_loader_line_augmented(
        data_path=train_cfg.data_path,
        bsz=100,
        n_max=train_cfg.n_max,
        min_depth=train_cfg.min_depth,
        num_workers=0,
        extra_embedding=train_cfg.extra_embedding,
        shuffle=False # Don't shuffle to see initial data
    )
    
    ys = []
    for i in range(10):
        try:
            x, y, z = next(train_loader)
            ys.append(y)
        except StopIteration:
            break
            
    all_y = torch.cat(ys)
    print(f"Loaded {len(all_y)} samples.")
    print(f"Mean y: {all_y.mean().item()}")
    print(f"Std y: {all_y.std().item()}")
    print(f"Min y: {all_y.min().item()}")
    print(f"Max y: {all_y.max().item()}")
    print(f"Fraction of 0.5: {(all_y == 0.5).float().mean().item()}")

if __name__ == "__main__":
    inspect_data()


