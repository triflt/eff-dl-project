import os
import io
import re
import zipfile
from collections import Counter

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from lstm import LSTMClassifier
# from lsq import QuantLinear, LinearInt
# from pact import QuantLinear, LinearInt
# from adaround import QuantLinear, LinearInt
from apot import QuantLinear, LinearInt

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "./data"
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
BATCH_SIZE = 32

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

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


PAD, UNK = "<pad>", "<unk>"


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


# ----------------------------
# Dataset with dynamic padding
# ----------------------------
class SMSDataset(Dataset):
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0
    for inputs, lengths, labels in dataloader:
        inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)

    return epoch_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    for inputs, lengths, labels in dataloader:
        if inputs.shape[0] < 16:
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

all_tokens = [tokenize(txt) for _, txt in data]
stoi, itos = build_vocab(all_tokens)
pad_idx = stoi[PAD]
vocab_size = len(itos)
print(f"Vocab size: {vocab_size}")

# Datasets
ds_train = SMSDataset(train_set, stoi)
ds_test  = SMSDataset(test_set, stoi)

# Loaders
collate = lambda b: collate_batch(b, pad_idx)
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
dl_test  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=64,
    hidden_dim=64,
    num_classes=2,
    pad_idx=pad_idx,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

train_loss = train_epoch(model, dl_train, criterion, optimizer, device)
val_loss, val_acc = evaluate(model, dl_test, criterion, device)
print(f"Epoch {1:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.1f}%")
print('\n\n\n')


model_qat = model.to_qat(bits=8, qat_linear_class=QuantLinear)
optimizer_qa = torch.optim.Adam(model_qat.parameters(), lr=1e-2)

for i in range(5):
    train_loss = train_epoch(model_qat, dl_train, criterion, optimizer_qa, device)
    val_loss, val_acc = evaluate(model_qat, dl_test, criterion, device)
    print(f"Epoch {i:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100}%")

    model_qantized = model_qat.quantize(bits=8, linear_int_class=LinearInt)
    model_qantized.to('cpu')
    val_loss, val_acc = evaluate(model_qantized, dl_test, criterion, 'cpu')
    print(f"Epoch {i:02d} | val_loss={val_loss:.4f} | val_acc={val_acc*100}%")
