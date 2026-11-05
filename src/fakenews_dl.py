"""Deep-learning helpers for PCS26/84 (LSTM / BiLSTM text classifiers, PyTorch).

Design notes
------------
* Padding is handled with packed sequences, so the final hidden state is the
  state at the LAST REAL TOKEN - sequence length (padding amount) cannot leak
  into the classifier.
* Vocabulary is built from training text only.
* set_seed() must be called BEFORE a model is constructed.
* Checkpoints store the weights, vocabulary, architecture, preprocessing spec
  and label mapping, and are loaded with torch.load(weights_only=True).
"""
from __future__ import annotations

import os
import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

PAD_ID, UNK_ID = 0, 1
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|[^\w\s]")


# ---------------------------------------------------------------- reproducibility
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- text -> ids
def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def build_vocab(train_texts, max_vocab: int = 20_000, min_freq: int = 1) -> dict:
    """Vocabulary from TRAINING texts only. 0=<PAD>, 1=<UNK>."""
    counts = Counter()
    for t in train_texts:
        counts.update(tokenize(t))
    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for word, c in counts.most_common(max_vocab - 2):
        if c >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int):
    """Returns (ids padded to max_len, true_length>=1). Head truncation."""
    ids = [vocab.get(tok, UNK_ID) for tok in tokenize(text)[:max_len]]
    if not ids:                      # packing needs length >= 1
        ids = [UNK_ID]
    length = len(ids)
    ids = ids + [PAD_ID] * (max_len - length)
    return ids, length


class EncodedDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        enc = [encode(t, vocab, max_len) for t in texts]
        self.ids = torch.tensor([e[0] for e in enc], dtype=torch.long)
        self.lengths = torch.tensor([e[1] for e in enc], dtype=torch.long)
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.lengths[i], self.labels[i]


def make_loader(dataset, batch_size=32, shuffle=False, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=gen if shuffle else None)


# ---------------------------------------------------------------- model
class RecurrentClassifier(nn.Module):
    """Embedding -> (Bi)LSTM over packed sequence -> dropout -> linear."""

    def __init__(self, vocab_size, embedding_dim=64, hidden_size=64, num_classes=2,
                 dropout=0.3, bidirectional=False, padding_idx=PAD_ID):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True,
                           bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.rnn(packed)          # h: (num_dirs, batch, hidden), original batch order
        feat = torch.cat((h[-2], h[-1]), dim=1) if self.bidirectional else h[-1]
        return self.classifier(self.dropout(feat))


# ---------------------------------------------------------------- train / eval
def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=1.0):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for ids, lengths, y in loader:
            ids, lengths, y = ids.to(device), lengths.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(ids, lengths)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += y.size(0)
    return {"loss": total_loss / total_n, "accuracy": total_correct / total_n}


@torch.no_grad()
def predict_proba(model, loader, device):
    """Returns (P(Real), true labels) as numpy arrays."""
    model.eval()
    probs, labels = [], []
    for ids, lengths, y in loader:
        logits = model(ids.to(device), lengths.to(device))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, vocab, config, history, best_epoch, val_metrics):
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "vocab": vocab,
        "config": config,
        "history": history,
        "best_epoch": int(best_epoch),
        "val_metrics": val_metrics,
    }, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = ck["config"]
    model = RecurrentClassifier(
        vocab_size=cfg["vocab_size"], embedding_dim=cfg["embedding_dim"],
        hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck["vocab"], cfg, ck


@torch.no_grad()
def predict_text(model, vocab, cfg, model_input: str, device="cpu"):
    """P(Real) for ONE already-preprocessed string (same builder as training)."""
    ids, length = encode(model_input, vocab, cfg["max_len"])
    logits = model(torch.tensor([ids], device=device),
                   torch.tensor([length], device=device))
    return float(torch.softmax(logits, dim=1)[0, 1])
