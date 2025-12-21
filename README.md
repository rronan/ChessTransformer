# ChessTransformer

A transformer similar to GPT2-124M trained to predict Stockfish evaluation and best move, on 300M chess positions. 

Bot available to play against here: https://lichess.org/@/alphaminustwo. No tree search, just sampling in the predicted move distribution, among legal moves.

## Installation

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Install the project dependencies:

```bash
uv sync
```

## Dataset

Dataset consists in approx. 300M chess positions (https://www.kaggle.com/datasets/lichess/chess-evaluations) along with stockfish evaluation.

Download the dataset:
```
wget https://database.lichess.org/lichess_db_eval.jsonl.zst
```

Uncompress it to `data/lichess_db_eval.jsonl`.

## Data Processing

### Input

The FEN is transformed into a `65x13` or `64x18` tensor, depending on dataprocessing. See `alphaminustwo/dataset.py` for details.

### Output

#### Evaluation:

Evaluation is the win probability for white, following the formula:

```
if y["mate"] is not None:
    return y["mate"] > 0
return 1 / (1 + math.exp(-0.00368208 * y["cp"]))
```

0 means black wins with proba 1
1 means white wins with proba 1

#### Best move:

Best move is encoded as a 64x64 one-hot representation of the starting and ending squares.

Implementation can be found in `alphaminustwo/dataset.py`.

## Model

The model is a transformer very similar to GPT2-124M, where the token embedding is replaced by a Linear layer of shape `13x768`. The language model head is replaced by two heads, MLPs, to predict the evaluation and best move.

Implementation can be found in `alphaminustwo/model.py`.

## Training:

We use Negative Log-Likelihood for evaluation (1D) and best move (64\*64D) prediction. We set the loss to `12 * evaluation_loss + move_loss`, so that both loss have the same scale (log2(64\*64) = 12). We use a batch size of `512`, linear warmup for `2000` steps and cosine annealing until the end of the training, at `600k` steps.

On validation set, we obtain a loss of `0.21` on evaluation and `1.56` on move prediction.

The curves look like this:

<img width="1302" height="590" alt="Screenshot 2025-12-21 at 19 24 29" src="https://github.com/user-attachments/assets/664a3dc2-8fcc-44c0-b861-a9f87f69d237" />

## Evaluation

### Puzzle

We evaluate on lichess puzzles, achieving a Elo of 1960

### Lichess bot

We evaluate on lichess.org, playing against human. The model seems to have approx. 1500 Elo in blitz and 1900 in bullet (probably a bit more in 1+0, a bit less in 1+1)

## Bot

Clone lichess-bot repository:
```
git clone https://github.com/lichess-bot-devs/lichess-bot
cd lichess-bot
```

In `homemade.py` import the bot:
```
from alphaminus.bot import AlphaMinusTwo
```

Edit `config.yml` to select the engine (see lichess-bot documentation) and run:
```
ALPHAMINUSTWO_CHKP=<path/to/checkpoint> python lichess-bot.py
```

## Thanks:
- lichess.org
- https://github.com/karpathy/nanoGPT: for weight initialization and optimizer configuration

See also this paper https://arxiv.org/abs/2402.04494 for a bigger model trained on a bigger dataset.
