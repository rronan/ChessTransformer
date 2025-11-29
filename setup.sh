
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync

wget https://database.lichess.org/lichess_db_eval.jsonl.zst
zstd -d lichess_db_eval.jsonl.zst -o data/lichess_db_eval.jsonl

wget https://database.lichess.org/lichess_db_puzzle.csv.zst
zstd -d lichess_db_puzzle.csv.zst -o data/lichess_db_puzzle.csv
