import torch
from alphaminustwo.model import GPT
from alphaminustwo import config

def debug_initialization():
    cfg = config.GPT124M()
    # Ensure extra_embedding matches what we think (False)
    print(f"Extra embedding: {cfg.extra_embedding}")
    
    model = GPT(cfg)
    
    # Check score_head dimensions
    first_linear = model.score_head[0]
    print(f"Score head first linear: {first_linear}")
    print(f"Weight shape: {first_linear.weight.shape}")
    print(f"Weight std: {first_linear.weight.std().item()}")
    
    # Create input
    bsz = 16
    x = torch.randint(0, 10, (bsz, 64, cfg.square_dim)).float() # Mock inputs roughly
    
    # Manually run part of forward to get input to score_head
    # We need real embeddings though to see activation stats
    # So let's just run the whole model
    
    # Mock valid inputs for embedding layers
    # wte takes (bsz, 64, square_dim) - float
    # wpe takes indices
    
    # Let's use random normal for x to simulate embeddings coming out of transformer
    # Input to score_head is (B, 64*n_embd)
    # Assuming transformer outputs unit variance-ish
    transformer_out_flat = torch.randn(bsz, 64 * cfg.n_embd)
    
    # Pass through first linear of score_head
    hidden = first_linear(transformer_out_flat)
    print(f"Hidden layer output mean: {hidden.mean().item()}")
    print(f"Hidden layer output std: {hidden.std().item()}")
    
    # Pass through GELU
    gelu_out = model.score_head[1](hidden)
    print(f"GELU output mean: {gelu_out.mean().item()}")
    print(f"GELU output std: {gelu_out.std().item()}")
    
    # Final score
    y_score = model.score_head[2](gelu_out)
    print(f"Final logits mean: {y_score.mean().item()}")
    print(f"Final logits std: {y_score.std().item()}")
    
    # Check sigmoid of logits (prediction probability)
    probs = torch.sigmoid(y_score)
    print(f"Probabilities mean: {probs.mean().item()}")
    print(f"Probabilities std: {probs.std().item()}")
    print(f"Min prob: {probs.min().item()}")
    print(f"Max prob: {probs.max().item()}")

if __name__ == "__main__":
    debug_initialization()

