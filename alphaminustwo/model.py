import inspect
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
import chess

from alphaminustwo.dataset import uci2index, index2uci, fen2tensor


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qhv = self.c_attn(x)
        q, k, v = torch.split(qhv, C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight_loss_move = config.weight_loss_move
        self.weight_loss_score = config.weight_loss_score
        self.extra_embedding = config.extra_embedding
        self.block_size = 65 if config.extra_embedding else 64
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.score_loss = config.score_loss
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(config.square_dim, self.n_embd),
                wpe=nn.Embedding(self.block_size, self.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(self.n_layer)]),
                ln_f=nn.LayerNorm(self.n_embd, bias=config.bias),
            )
        )
        # differs from gpt
        self.score_head = nn.Sequential(
            nn.Linear(
                self.n_embd * (1 if self.extra_embedding else 64),
                self.n_embd,
                bias=config.bias,
            ),
            nn.GELU(),
            nn.Linear(self.n_embd, 1, bias=config.bias),
        )
        self.move_head = nn.Sequential(
            nn.Linear(
                self.n_embd * (64 if self.extra_embedding else 1),
                self.n_embd,
                bias=config.bias,
            ),
            nn.GELU(),
            nn.Linear(
                self.n_embd, 64 * (64 if self.extra_embedding else 1), bias=config.bias
            ),
        )
        self.apply(self._init_weights)

    def forward(
        self,
        x,
        score: Optional[torch.Tensor] = None,
        move: Optional[torch.Tensor] = None,
    ):
        p = torch.arange(0, self.block_size, dtype=torch.long, device=x.device)
        x = self.transformer.wte(x) + self.transformer.wpe(p)
        for h in self.transformer.h:
            x = h(x)
        x = self.transformer.ln_f(x)
        if self.extra_embedding:
            y_score = self.score_head(x[:, 0]).view(-1)
            y_move = self.move_head(x[:, 1:].view(-1, 64 * self.n_embd)).view(-1, 64**2)
        else:
            y_score = self.score_head(x.view(-1, 64 * self.n_embd)).view(-1)
            y_move = self.move_head(x).view(-1, 64**2)
        loss_score = None
        if score is not None:
            if self.score_loss == "bce":
                loss_score = F.binary_cross_entropy_with_logits(y_score, score)
            elif self.score_loss == "mse":
                loss_score = F.mse_loss(y_score, score)
        loss_move = None
        if move is not None:
            loss_move = F.cross_entropy(y_move, move)
        loss = None
        if loss_score is not None and loss_move is not None:
            loss = (
                loss_score * self.weight_loss_score + loss_move * self.weight_loss_move
            )
        return y_score, y_move, loss_score, loss_move, loss

    def generate_from_board(self, board_list: list, legal_move: bool = False):
        x_list = [
            fen2tensor(board.fen(), extra_embedding=self.extra_embedding)
            for board in board_list
        ]
        x = torch.stack(x_list).to(self.device())
        with torch.no_grad():
            score, logits, *_ = self.forward(x, None)
        if legal_move:
            for i, board in enumerate(board_list):
                mask = torch.zeros(64**2).to(self.device())
                for move in board.legal_moves:
                    mask[uci2index(move.uci())] = 1
                logits[i].masked_fill_(mask == 0, float("-inf"))
        index_batch = torch.multinomial(logits.exp(), 1)
        res = [chess.Move.from_uci(index2uci(index.item())) for index in index_batch]
        return res, score.tolist()

    def _init_weights(self, module):
        """
        From: https://github.com/karpathy/nanoGPT/blob/master/model.py
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        From here: https://github.com/karpathy/nanoGPT/blob/master/model.py
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def device(self):
        return next(self.parameters()).device
