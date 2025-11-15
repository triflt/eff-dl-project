"""
Interactive inference utility that loads the tokenizer and either the base
or quantized LSTM classifier checkpoints saved by main_train.py.
"""

import argparse
import json
import re
from pathlib import Path

import torch

# Ensure the custom modules referenced by the pickled checkpoints are registered.
from lstm import LSTMClassifier  # noqa: F401
from efficientqat import LinearInt  # noqa: F401

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
PAD, UNK = "<pad>", "<unk>"

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_BASE_MODEL_PATH = ARTIFACT_DIR / 'base' / "lstm_base.pt"
DEFAULT_TOKENIZER_PATH = ARTIFACT_DIR / 'base' / "tokenizer.json"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def encode(tokens: list[str], stoi: dict[str, int], pad_idx: int) -> list[int]:
    unk_id = stoi.get(UNK)
    if unk_id is None:
        raise ValueError("Tokenizer dictionary is missing the <unk> token.")
    encoded = [stoi.get(tok, unk_id) for tok in tokens]
    if not encoded:
        encoded = [pad_idx]
    return encoded


def load_tokenizer(path: Path) -> tuple[dict[str, int], int]:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "stoi" not in payload or "pad_idx" not in payload:
        raise ValueError("Tokenizer payload is missing required keys.")

    stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
    pad_idx = int(payload["pad_idx"])
    return stoi, pad_idx


def run_inference(model: LSTMClassifier, stoi: dict[str, int], pad_idx: int, text: str, device: torch.device) -> tuple[int, torch.Tensor]:
    tokens = tokenize(text)
    ids = encode(tokens, stoi, pad_idx)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_tensor, lengths)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
    return int(pred.item()), probs.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser(description="Run interactive inference with a saved LSTM classifier.")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Optional path to the checkpoint. Defaults to artifacts/lstm_base.pt or lstm_quantized.pt.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="Path to the saved tokenizer JSON.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(args.model_path)
    if args.model_path is not None:
        model_path = ARTIFACT_DIR / args.model_path / "lstm_quantized.pt"
    else:
        model_path = DEFAULT_BASE_MODEL_PATH
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    stoi, pad_idx = load_tokenizer(args.tokenizer_path)

    model = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(model, LSTMClassifier):
        raise TypeError(f"Loaded checkpoint is not an LSTMClassifier (got {type(model)!r}).")
    model.eval()
    print(model)
    model.to(device)

    print(f"Loaded model from {model_path}")
    print("Press Enter on an empty line to exit.")

    while True:
        try:
            text = input("Enter text: ").strip()
        except EOFError:
            break
        if not text:
            break
        pred_idx, probs = run_inference(model, stoi, pad_idx, text, device)
        label = "positive" if pred_idx == 1 else "negative"
        confidence = probs[pred_idx].item()
        print(f"Prediction: {label} ({confidence:.2%} confidence)\n")


if __name__ == "__main__":
    main()
