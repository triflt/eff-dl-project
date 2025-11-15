import os
import zipfile
import io
import json
import re
import time
from collections import Counter
from pathlib import Path

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from lstm import LSTMClassifier
# from lsq import QuantLinear, LinearInt
# from pact import QuantLinear, LinearInt
# from adaround import QuantLinear, LinearInt
# from apot import QuantLinear, LinearInt
from efficientqat import QuantLinear, LinearInt

torch.manual_seed(0)
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
BATCH_SIZE = 64
BASE_EPOCHS = 1
QAT_EPOCHS = 2

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "./data"
PAD, UNK = "<pad>", "<unk>"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "eq"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
BASE_MODEL_PATH = ARTIFACT_DIR / "lstm_base.pt"
QUANT_MODEL_PATH = ARTIFACT_DIR / "lstm_quantized.pt"
TOKENIZER_PATH = ARTIFACT_DIR / "tokenizer.json"
BASE_LOSS_PLOT_PATH = ARTIFACT_DIR / "base_loss.png"
QAT_LOSS_PLOT_PATH = ARTIFACT_DIR / "qat_loss.png"
TIMINGS_PATH = ARTIFACT_DIR / "timings.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


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

data = download_sms_dataset()
# Tokenize once to build vocab on train only
train_p = 0.7
train_set, test_set = random_split(data, lengths=(train_p, 1 - train_p))

all_tokens = [tokenize(txt) for _, txt in train_set]
stoi, itos = build_vocab(all_tokens)
pad_idx = stoi[PAD]
vocab_size = len(itos)
print(f"Vocab size: {vocab_size}")
tokenizer_payload = {
    "stoi": stoi,
    "itos": itos,
    "pad_idx": pad_idx,
    "pad_token": PAD,
    "unk_token": UNK,
}
with TOKENIZER_PATH.open("w", encoding="utf-8") as f:
    json.dump(tokenizer_payload, f, ensure_ascii=True, indent=2)
print(f"Saved tokenizer to {TOKENIZER_PATH}")

# Datasets
ds_train = TextDataset(train_set, stoi)
ds_test  = TextDataset(test_set, stoi)

# Loaders
collate = lambda b: collate_batch(b, pad_idx)
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
dl_test  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=96,
    hidden_dim=96,
    num_classes=2,
    num_layers=2,
    pad_idx=pad_idx,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
base_total_steps = max(1, BASE_EPOCHS * len(dl_train))
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=base_total_steps)
base_iter_losses: list[float] = []
qat_iter_losses: list[float] = []
timings = {
    "train": 0.0,
    "infer": 0.0,
    "qat_train": 0.0,
    "qat_infer": 0.0,
    "quantized_infer": 0.0,
}
metrics = {
    "train_acc": [],
    "qat_acc": [],
    "quantized_acc": [],
}

for epoch in range(1, BASE_EPOCHS + 1):
    start_time = time.perf_counter()
    train_loss, batch_losses = train_epoch(model, dl_train, criterion, optimizer, device, scheduler=base_scheduler)
    timings["train"] += time.perf_counter() - start_time
    base_iter_losses.extend(batch_losses)

    start_time = time.perf_counter()
    val_loss, val_acc = evaluate(model, dl_test, criterion, device)
    timings["infer"] += time.perf_counter() - start_time
    metrics["train_acc"].append(val_acc)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.1f}%")
    print("\n")

base_train_steps = list(range(1, len(base_iter_losses) + 1))
save_loss_plot(base_iter_losses, base_train_steps, BASE_LOSS_PLOT_PATH, "Base Model Loss")

model.eval()
torch.save(model.cpu(), BASE_MODEL_PATH)
model.to(device)
print(f"Saved base model checkpoint to {BASE_MODEL_PATH}")
model.train()

model_qat = model.to_qat(bits=8, qat_linear_class=QuantLinear)
del model
torch.cuda.empty_cache()
optimizer_qa = torch.optim.AdamW(model_qat.parameters(), lr=5e-3)
qat_total_steps = max(1, QAT_EPOCHS * len(dl_train))
qat_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_qa, T_max=qat_total_steps)
final_quantized_model = None

for epoch in range(1, QAT_EPOCHS + 1):
    start_time = time.perf_counter()
    train_loss, batch_losses = train_epoch(
        model_qat, dl_train, criterion, optimizer_qa, device, scheduler=qat_scheduler
    )
    timings["qat_train"] += time.perf_counter() - start_time
    qat_iter_losses.extend(batch_losses)

    start_time = time.perf_counter()
    val_loss, val_acc = evaluate(model_qat, dl_test, criterion, device)
    timings["qat_infer"] += time.perf_counter() - start_time
    metrics["qat_acc"].append(val_acc)
    print(f"QAT Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.1f}%")

    model_qantized = model_qat.quantize(bits=8, linear_int_class=LinearInt)
    quant_eval_device = device
    model_qantized.to(quant_eval_device)
    start_time = time.perf_counter()
    val_loss, val_acc = evaluate(model_qantized, dl_test, criterion, quant_eval_device)
    timings["quantized_infer"] += time.perf_counter() - start_time
    metrics["quantized_acc"].append(val_acc)
    print(f"Quantized Epoch {epoch:02d} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.1f}%")
    final_quantized_model = model_qantized

if qat_iter_losses:
    qat_train_steps = list(range(1, len(qat_iter_losses) + 1))
    save_loss_plot(qat_iter_losses, qat_train_steps, QAT_LOSS_PLOT_PATH, "QAT Model Loss")

if final_quantized_model is not None:
    final_quantized_model.eval()
    final_quantized_model.to('cpu')
    torch.save(final_quantized_model, QUANT_MODEL_PATH)
    print(f"Saved quantized model checkpoint to {QUANT_MODEL_PATH}")

with TIMINGS_PATH.open("w", encoding="utf-8") as f:
    json.dump(timings, f, ensure_ascii=True, indent=2)
print(f"Saved timing information to {TIMINGS_PATH}")

with METRICS_PATH.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=True, indent=2)
print(f"Saved metrics information to {METRICS_PATH}")
