
# If uv not installed: 
# curl -LsSf https://astral.sh/uv/install.sh | sh
# then add vu to PATH

uv sync

mkdir -p data

wget https://database.lichess.org/lichess_db_eval.jsonl.zst
zstd -d lichess_db_eval.jsonl.zst -o data/lichess_db_eval.jsonl

wget https://database.lichess.org/lichess_db_puzzle.csv.zst
zstd -d lichess_db_puzzle.csv.zst -o data/lichess_db_puzzle.csv
