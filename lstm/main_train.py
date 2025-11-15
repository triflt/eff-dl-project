import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm.auto import tqdm

from lstm import LSTMClassifier
# from lsq import QuantLinear, LinearInt
# from pact import QuantLinear, LinearInt
# from adaround import QuantLinear, LinearInt
# from apot import QuantLinear, LinearInt
from efficientqat import QuantLinear, LinearInt

MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
BATCH_SIZE = 32

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


class TextDataset(Dataset):
    def __init__(self, data, stoi):
        self.labels = data['label']
        self.texts = [encode(tokenize(x), stoi) for x in data['text']]

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
    for inputs, lengths, labels in tqdm(dataloader):
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
    for inputs, lengths, labels in tqdm(dataloader):
        if inputs.shape[0] < 16:
            continue
        inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        epoch_loss += loss.item() * inputs.size(0)
        epoch_acc += accuracy(logits, labels) * inputs.size(0)

    dataset_size = len(dataloader.dataset)
    return epoch_loss / dataset_size, epoch_acc / dataset_size

train_set, test_set = load_dataset("stanfordnlp/imdb", split=['train', 'test'])

all_tokens = [tokenize(txt) for txt in train_set['text']]
stoi, itos = build_vocab(all_tokens)
pad_idx = stoi[PAD]
vocab_size = len(itos)
print(f"Vocab size: {vocab_size}")

# Datasets
ds_train = TextDataset(train_set[:10_000], stoi)
ds_test  = TextDataset(test_set[:10_000], stoi)

# Loaders
collate = lambda b: collate_batch(b, pad_idx)
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
dl_test  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=48,
    hidden_dim=48,
    num_classes=2,
    pad_idx=pad_idx,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_loss = train_epoch(model, dl_train, criterion, optimizer, device)
val_loss, val_acc = evaluate(model, dl_test, criterion, device)
print(f"Epoch {1:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.1f}%")
print('\n')


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
