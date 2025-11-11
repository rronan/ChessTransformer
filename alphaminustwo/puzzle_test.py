import tempfile
import csv
from alphaminustwo.model import GPT
from alphaminustwo.config import GPT124M

from alphaminustwo.puzzle import load_puzzles, evaluate_puzzle


def test_load_puzzles():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "PuzzleId",
                "FEN",
                "Moves",
                "Rating",
                "RatingDeviation",
                "Popularity",
                "NbPlays",
                "Themes",
                "GameUrl",
                "OpeningTags",
            ]
        )
        writer.writerow(
            [
                "test1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
                "e2e4 e7e5",
                "1500",
                "75",
                "90",
                "100",
                "opening",
                "https://lichess.org/test",
                "",
            ]
        )
        writer.writerow(
            [
                "test2",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
                "d2d4 d7d5",
                "1600",
                "75",
                "90",
                "100",
                "opening",
                "https://lichess.org/test2",
                "",
            ]
        )
        writer.writerow(
            [
                "test3",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
                "c2c4 c7c5",
                "800",
                "75",
                "90",
                "100",
                "opening",
                "https://lichess.org/test3",
                "",
            ]
        )
        csv_path = f.name
    puzzles = load_puzzles(csv_path)
    assert len(puzzles) == 3
    puzzles = load_puzzles(csv_path, max_puzzles=2)
    assert len(puzzles) == 2
    valid_ids = {"test1", "test2", "test3"}
    for puzzle in puzzles:
        assert puzzle["id"] in valid_ids


def test_evaluate_puzzle():
    """Test puzzle evaluation returns correct structure."""
    model_cfg = GPT124M()
    model = GPT(model_cfg)
    model.eval()
    puzzle = {
        "id": "test_puzzle",
        "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
        "moves": ["f2g3", "e6e7", "b2b1", "b3c1"],
        "rating": 1743,
        "themes": "crushing",
    }
    evaluate_puzzle(model, puzzle)
