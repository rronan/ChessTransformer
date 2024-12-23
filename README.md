# ChessTransformer

A transformer similar to GPT2-124M trained to predict Stockfish evaluation and best move, on 132M chess positions. 

Bot available to play against here: https://lichess.org/@/alphaminustwo. No tree search, just sampling in the predicted move distribution, among legal moves.

Thanks:
- lichess.org
- https://github.com/karpathy/nanoGPT
- https://www.kaggle.com/datasets/lichess/chess-evaluations
