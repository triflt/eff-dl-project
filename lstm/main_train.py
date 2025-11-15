import os
import zipfile
import io
import json
import re
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Optional

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from lstm import LSTMClassifier
from lsq import QuantLinear as QLLSQ, LinearInt as LILSQ
from pact import QuantLinear as QLPACT, LinearInt as LIPACT
from adaround import QuantLinear as QLA, LinearInt as LIA
from apot import QuantLinear as QLAP, LinearInt as LIAP
from efficientqat import QuantLinear as QLEF, LinearInt as LIEF

qat_dict = {
    "lsq": {"ql": QLLSQ, "li": LILSQ},
    "pact": {"ql": QLPACT, "li": LIPACT},
    "adaround": {"ql": QLA, "li": LIA},
    "apot": {"ql": QLAP, "li": LIAP},
    "efficientqat": {"ql": QLEF, "li": LIEF},
}

torch.manual_seed(0)
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
BATCH_SIZE = 64
TRAIN_RATIO = 0.7

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "./data"
PAD, UNK = "<pad>", "<unk>"


def download_sms_dataset() -> list[tuple[int, str]]:
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "smsspam.zip")
    txt_path = os.path.join(DATA_DIR, "SMSSpamCollection")

    if not os.path.exists(txt_path):
        print("Downloading dataset...")
        r = requests.get(DATA_URL, timeout=30)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
    else:
        print("Dataset already present.")

    data = []
    with io.open(txt_path, encoding="utf-8") as f:
        for line in f:
            label, text = line.strip().split("\t", 1)
            y = 1 if label == "spam" else 0
            data.append((y, text))
    return data


def build_vocab(texts: list[list[str]]):
    counter = Counter()
    for toks in texts:
        counter.update(toks)
    # keep by freq, cap by size
    vocab = [PAD, UNK]
    for w, c in counter.most_common():
        if c < MIN_FREQ: break
        vocab.append(w)
        if len(vocab) >= MAX_VOCAB_SIZE: break
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = vocab
    return stoi, itos


def encode(tokens: list[str], stoi: dict) -> list[int]:
    unk = stoi.get(UNK)
    return [stoi.get(t, unk) for t in tokens]


class TextDataset(Dataset):
    def __init__(self, samples, stoi):
        self.labels = [y for y, _ in samples]
        self.texts = [encode(tokenize(x), stoi) for _, x in samples]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_batch(batch, pad_idx):
    # batch: list of (tensor_ids, label)
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    # sort by length desc for pack_padded_sequence
    lengths, sort_idx = lengths.sort(descending=True)
    padded = padded[sort_idx]
    labels = torch.stack(labels)[sort_idx]
    return padded, lengths, labels


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct / targets.size(0)


def save_loss_plot(
    train_losses: list[float],
    train_steps: list[int],
    path: Path,
    title: str,
) -> None:
    plt.figure()
    plt.plot(train_steps, train_losses, label="Train")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {title} plot to {path}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler = None,
) -> tuple[float, list[float]]:
    model.train()
    epoch_loss = 0.0
    batch_losses: list[float] = []
    for inputs, lengths, labels in tqdm(dataloader):
        inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_item = loss.item()
        batch_losses.append(loss_item)
        epoch_loss += loss_item * inputs.size(0)

    return epoch_loss / len(dataloader.dataset), batch_losses


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    for inputs, lengths, labels in tqdm(dataloader):
        if inputs.shape[0] <= 16:
            continue
        inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        epoch_loss += loss.item() * inputs.size(0)
        epoch_acc += accuracy(logits, labels) * inputs.size(0)

    dataset_size = len(dataloader.dataset)
    return epoch_loss / dataset_size, epoch_acc / dataset_size


def prepare_dataset(train_ratio: float = TRAIN_RATIO) -> tuple[list[tuple[int, str]], list[tuple[int, str]], dict[str, Any]]:
    """Download the dataset (if needed) and build cached train/test splits plus tokenizer."""
    data = download_sms_dataset()
    if len(data) < 2:
        raise RuntimeError("Dataset must contain at least 2 samples to create train/test splits.")

    train_len = max(1, int(len(data) * train_ratio))
    test_len = len(data) - train_len
    if test_len == 0:
        test_len = 1
        train_len = len(data) - test_len

    generator = torch.Generator().manual_seed(0)
    train_subset, test_subset = random_split(data, [train_len, test_len], generator=generator)
    train_samples = list(train_subset)
    test_samples = list(test_subset)

    all_tokens = [tokenize(txt) for _, txt in train_samples]
    stoi, itos = build_vocab(all_tokens)
    pad_idx = stoi[PAD]
    print(f"Vocab size: {len(itos)}")
    tokenizer_payload = {
        "stoi": stoi,
        "itos": itos,
        "pad_idx": pad_idx,
        "pad_token": PAD,
        "unk_token": UNK,
    }
    return train_samples, test_samples, tokenizer_payload


def build_dataloaders(
    train_samples: list[tuple[int, str]],
    test_samples: list[tuple[int, str]],
    tokenizer: dict[str, Any],
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test dataloaders for the cached samples."""
    stoi = tokenizer["stoi"]
    pad_idx = tokenizer["pad_idx"]
    collate_fn = lambda batch: collate_batch(batch, pad_idx)
    ds_train = TextDataset(train_samples, stoi)
    ds_test = TextDataset(test_samples, stoi)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dl_train, dl_test

def train_base_model(
    train_samples,
    test_samples,
    tokenizer,
    batch_size: int,
    lr: float,
    lstm_dim: int,
    num_layers: int,
    epochs: int,
) -> tuple[nn.Module, dict[str, Any], dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    """Train the floating-point LSTM model and return the model plus training artifacts."""
    dl_train, dl_test = build_dataloaders(train_samples, test_samples, tokenizer, batch_size)

    vocab_size = len(tokenizer["itos"])
    pad_idx = tokenizer["pad_idx"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=lstm_dim,
        hidden_dim=lstm_dim,
        num_classes=2,
        num_layers=num_layers,
        pad_idx=pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(1, epochs * len(dl_train))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    timings = {"train": 0.0, "infer": 0.0}
    metrics = {"train_acc": []}
    losses = {"train": []}

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        train_loss, batch_losses = train_epoch(model, dl_train, criterion, optimizer, device, scheduler=scheduler)
        timings["train"] += time.perf_counter() - start_time
        losses["train"].extend(batch_losses)

        start_time = time.perf_counter()
        val_loss, val_acc = evaluate(model, dl_test, criterion, device)
        timings["infer"] += time.perf_counter() - start_time
        metrics["train_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.1f}%"
        )

    return model, tokenizer, timings, metrics, losses


def train_qat_models(
    train_samples,
    test_samples,
    tokenizer,
    base_model: nn.Module,
    batch_size: int,
    lr: float,
    epochs: int,
    qat_method: str,
) -> tuple[nn.Module, Optional[nn.Module], dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    """Fine-tune the base model with QAT and additionally produce a quantized model."""
    dl_train, dl_test = build_dataloaders(train_samples, test_samples, tokenizer, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model_qat = base_model.to(device).to_qat(bits=8, qat_linear_class=qat_dict[qat_method]['ql'])
    optimizer = torch.optim.AdamW(model_qat.parameters(), lr=lr)
    total_steps = max(1, epochs * len(dl_train))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    timings = {"qat_train": 0.0, "qat_infer": 0.0, "quantized_infer": 0.0}
    metrics = {"qat_acc": [], "quantized_acc": []}
    losses = {"qat_train": [], "quantized_eval": []}
    final_quantized_model: Optional[nn.Module] = None

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        train_loss, batch_losses = train_epoch(
            model_qat, dl_train, criterion, optimizer, device, scheduler=scheduler
        )
        timings["qat_train"] += time.perf_counter() - start_time
        losses["qat_train"].extend(batch_losses)

        start_time = time.perf_counter()
        val_loss, val_acc = evaluate(model_qat, dl_test, criterion, device)
        timings["qat_infer"] += time.perf_counter() - start_time
        metrics["qat_acc"].append(val_acc)
        print(
            f"QAT Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.1f}%"
        )

        quantized_model = model_qat.quantize(bits=8, linear_int_class=qat_dict[qat_method]['li'])
        quantized_model.to(device)
        start_time = time.perf_counter()
        q_val_loss, q_val_acc = evaluate(quantized_model, dl_test, criterion, device)
        timings["quantized_infer"] += time.perf_counter() - start_time
        metrics["quantized_acc"].append(q_val_acc)
        losses["quantized_eval"].append(q_val_loss)
        print(f"Quantized Epoch {epoch:02d} | val_loss={q_val_loss:.4f} | val_acc={q_val_acc*100:.1f}%")
        final_quantized_model = quantized_model

    return final_quantized_model, timings, metrics, losses


def _save_model_checkpoint(model: nn.Module, path: Path) -> None:
    """Persist a model on CPU while keeping it on the original device afterwards."""
    original_device: Optional[torch.device] = None
    first_param = next(model.parameters(), None)
    if first_param is not None:
        original_device = first_param.device

    model_cpu = model.to("cpu")
    torch.save(model_cpu, path)
    print(f"Saved model checkpoint to {path}")
    if original_device is not None:
        model.to(original_device)


def _prepare_artifact_dir(name: str) -> Path:
    artifact_dir = Path(__file__).resolve().parent / "artifacts" / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _merge_and_save_json(path: Path, payload: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
    existing.update(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=True, indent=2)


def save_base_artifacts(
    name: str,
    model: Optional[nn.Module],
    tokenizer: Optional[dict[str, Any]],
    timings: dict[str, float],
    metrics: dict[str, list[float]],
    losses: dict[str, list[float]],
) -> None:
    artifact_dir = _prepare_artifact_dir(name)
    base_model_path = artifact_dir / "lstm_base.pt"
    tokenizer_path = artifact_dir / "tokenizer.json"
    base_loss_plot_path = artifact_dir / "base_loss.png"
    timings_path = artifact_dir / "timings.json"
    metrics_path = artifact_dir / "metrics.json"

    if tokenizer is not None:
        with tokenizer_path.open("w", encoding="utf-8") as f:
            json.dump(tokenizer, f, ensure_ascii=True, indent=2)
        print(f"Saved tokenizer to {tokenizer_path}")

    if model is not None:
        _save_model_checkpoint(model.eval(), base_model_path)

    if losses.get("train"):
        base_steps = list(range(1, len(losses["train"]) + 1))
        save_loss_plot(losses["train"], base_steps, base_loss_plot_path, "Base Model Loss")

    _merge_and_save_json(timings_path, timings)
    print(f"Saved base timing information to {timings_path}")

    _merge_and_save_json(metrics_path, metrics)
    print(f"Saved base metrics information to {metrics_path}")


def save_qat_artifacts(
    name: str,
    quantized_model: Optional[nn.Module],
    timings: dict[str, float],
    metrics: dict[str, list[float]],
    losses: dict[str, list[float]],
) -> None:
    artifact_dir = _prepare_artifact_dir(name)
    quant_model_path = artifact_dir / "lstm_quantized.pt"
    qat_loss_plot_path = artifact_dir / "qat_loss.png"
    timings_path = artifact_dir / "timings.json"
    metrics_path = artifact_dir / "metrics.json"

    if quantized_model is not None:
        quantized_model.eval()
        quantized_model_cpu = quantized_model.to("cpu")
        torch.save(quantized_model_cpu, quant_model_path)
        print(f"Saved quantized model checkpoint to {quant_model_path}")

    if losses.get("qat_train"):
        qat_steps = list(range(1, len(losses["qat_train"]) + 1))
        save_loss_plot(losses["qat_train"], qat_steps, qat_loss_plot_path, "QAT Model Loss")

    _merge_and_save_json(timings_path, timings)
    print(f"Saved QAT timing information to {timings_path}")

    _merge_and_save_json(metrics_path, metrics)
    print(f"Saved QAT metrics information to {metrics_path}")


def main() -> None:
    """Train the base and QAT models end-to-end and save the resulting artifacts."""
    train_samples, test_samples, tokenizer = prepare_dataset()
    base_model, tokenizer, base_timings, base_metrics, base_losses = train_base_model(
        train_samples,
        test_samples,
        tokenizer,
        batch_size=BATCH_SIZE,
        lr=3e-4,
        lstm_dim=96,
        num_layers=2,
        epochs=1,
    )

    save_base_artifacts(
        name='base',
        model=base_model,
        tokenizer=tokenizer,
        timings=base_timings,
        metrics=base_metrics,
        losses=base_losses,
    )

    for name, epochs, qat_method, lr in [
        # ("lsq_1_5e-3", 1, "lsq", 5e-3),
        # ("lsq_2_5e-3", 2, "lsq", 5e-3),
        # ("lsq_1_1e-3", 1, "lsq", 1e-3),
        # ("lsq_2_1e-3", 2, "lsq", 1e-3),
        # ("lsq_1_1e-2", 1, "lsq", 1e-2),
        # ("lsq_2_1e-2", 2, "lsq", 1e-2),
        ("lsq_1_5e-2", 1, "lsq", 5e-2),
        ("lsq_2_5e-2", 2, "lsq", 5e-2),

        # ("pact_1_5e-3", 1, "pact", 5e-3),
        # ("pact_2_5e-3", 2, "pact", 5e-3),
        # ("pact_1_1e-3", 1, "pact", 1e-3),
        # ("pact_2_1e-3", 2, "pact", 1e-3),
        # ("pact_1_1e-2", 1, "pact", 1e-2),
        # ("pact_2_1e-2", 2, "pact", 1e-2),
        ("pact_1_5e-2", 1, "pact", 5e-2),
        ("pact_2_5e-2", 2, "pact", 5e-2),

        # ("adaround_1_5e-3", 1, "adaround", 5e-3),
        # ("adaround_2_5e-3", 2, "adaround", 5e-3),
        # ("adaround_1_1e-3", 1, "adaround", 1e-3),
        # ("adaround_2_1e-3", 2, "adaround", 1e-3),
        # ("adaround_1_1e-2", 1, "adaround", 1e-2),
        # ("adaround_2_1e-2", 2, "adaround", 1e-2),
        ("adaround_1_5e-2", 1, "adaround", 5e-2),
        ("adaround_2_5e-2", 2, "adaround", 5e-2),

        # ("apot_1_5e-3", 1, "apot", 5e-3),
        # ("apot_2_5e-3", 2, "apot", 5e-3),
        # ("apot_1_1e-3", 1, "apot", 1e-3),
        # ("apot_2_1e-3", 2, "apot", 1e-3),
        # ("apot_1_1e-2", 1, "apot", 1e-2),
        # ("apot_2_1e-2", 2, "apot", 1e-2),
        ("apot_1_5e-2", 1, "apot", 5e-2),
        ("apot_2_5e-2", 2, "apot", 5e-2),

        # ("efficientqat_1_5e-3", 1, "efficientqat", 5e-3),
        # ("efficientqat_2_5e-3", 2, "efficientqat", 5e-3),
        # ("efficientqat_1_1e-3", 1, "efficientqat", 1e-3),
        # ("efficientqat_2_1e-3", 2, "efficientqat", 1e-3),
        # ("efficientqat_1_1e-2", 1, "efficientqat", 1e-2),
        # ("efficientqat_2_1e-2", 2, "efficientqat", 1e-2),
        ("efficientqat_1_5e-2", 1, "efficientqat", 5e-2),
        ("efficientqat_2_5e-2", 2, "efficientqat", 5e-2),
    ]:
        quantized_model, qat_timings, qat_metrics, qat_losses = train_qat_models(
            train_samples,
            test_samples,
            tokenizer,
            base_model=base_model,
            batch_size=BATCH_SIZE,
            lr=lr,
            epochs=epochs,
            qat_method=qat_method,
        )

        save_qat_artifacts(
            name=name,
            quantized_model=quantized_model,
            timings=qat_timings,
            metrics=qat_metrics,
            losses=qat_losses,
        )


if __name__ == "__main__":
    main()
