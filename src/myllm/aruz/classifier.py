"""Neural metre classifier: a small character-level Transformer trained on Ganjoor's labels.

    python -m myllm.aruz.classifier train --smoke                       # seconds, CPU
    python -m myllm.aruz.classifier train --out out/aruz/classifier.pt  # ~1h on 4 CPU cores
    python -m myllm.aruz.classifier train --device cuda --epochs 6      # RTX 4060

Why: the rule-based scanner explains *why* a line has a metre, but it is fooled by words it
cannot vowelise. Hundreds of thousands of labelled hemistichs teach a network the statistical
shape of each metre directly from the letters. At inference the two are combined
(``Scanner(classifier=..., classifier_weight=β)``): the network votes, the scanner verifies and
explains, and disagreement lowers the confidence.

Training details
* input: characters incl. harakat and ZWNJ (``normalize_verse``); harakat are dropped at random
  during training (p=0.5) so the model also works on unvowelised text
* labels: ``meter_key`` (pattern, or one class for the rubāʿī family); classes with < 50 lines skipped
* balanced: at most ``--per-class-cap`` lines per class (Shahnameh would otherwise dominate)
* split: Ganjoor poems with ``is_holdout`` (20%) are the dev set; test poets are never loaded
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from myllm.aruz.phonology import MARKS, normalize_verse

PAD, UNK = 0, 1
MAX_LEN = 72


class MeterNet(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, d: int = 128, layers: int = 3, heads: int = 4,
                 ff: int = 384, dropout: float = 0.1, max_len: int = MAX_LEN):
        super().__init__()
        self.config = {"d": d, "layers": layers, "heads": heads, "ff": ff, "dropout": dropout}
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids == PAD
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        x = self.enc(x, src_key_padding_mask=mask)
        keep = (~mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        return self.head(self.norm(pooled))


def _clean(text: str, drop_marks: bool = False) -> str:
    t = normalize_verse(text)
    if drop_marks:
        t = "".join(ch for ch in t if ch not in MARKS)
    return t


class MeterClassifier:
    """Inference wrapper: ``logprobs(texts) -> [{meter_key: log p}, ...]``."""

    def __init__(self, model: MeterNet, vocab: dict[str, int], classes: list[str], device: str = "cpu"):
        self.model = model.to(device).eval()
        self.vocab, self.classes, self.device = vocab, classes, device

    def encode(self, texts: list[str], drop_marks: bool = False) -> torch.Tensor:
        rows = []
        for t in texts:
            ids = [self.vocab.get(ch, UNK) for ch in _clean(t, drop_marks)][:MAX_LEN] or [UNK]
            rows.append(ids + [PAD] * (MAX_LEN - len(ids)))
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def logprobs(self, texts: list[str]) -> list[dict[str, float]]:
        logits = self.model(self.encode(texts))
        lp = torch.log_softmax(logits.float(), -1).cpu().tolist()
        return [dict(zip(self.classes, row)) for row in lp]

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict(), "vocab": self.vocab, "classes": self.classes,
                    "config": self.model.config, "meta": meta or {}}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> MeterClassifier:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = MeterNet(len(ck["vocab"]) + 2, len(ck["classes"]), **ck["config"])
        model.load_state_dict(ck["state"])
        obj = cls(model, ck["vocab"], ck["classes"], device)
        obj.meta = ck.get("meta", {})
        return obj


def build_dataset(roots: list[str], holdout: float, per_class_cap: int, min_class: int, seed: int, smoke: bool):
    from myllm.aruz.ganjoor import load_corpus
    from myllm.aruz.lexicon import is_holdout
    from myllm.aruz.meters import meter_for_rhythm, meter_key

    rows = []
    for root in roots:
        rows.extend(load_corpus(Path(root)))
    rng = random.Random(seed)
    rng.shuffle(rows)
    labelled = []
    for h in rows:
        m = meter_for_rhythm(h.rhythm or "")
        if m is not None:
            labelled.append((h, meter_key(m)))
    counts = Counter(k for h, k in labelled if not is_holdout(h.poem_url, holdout))
    classes = sorted(k for k, n in counts.items() if n >= (5 if smoke else min_class))
    cidx = {k: i for i, k in enumerate(classes)}
    train, dev, per = [], [], Counter()
    cap = 40 if smoke else per_class_cap
    for h, k in labelled:
        if k not in cidx:
            continue
        if is_holdout(h.poem_url, holdout):
            dev.append((h.text, cidx[k]))
        elif per[k] < cap:
            per[k] += 1
            train.append((h.text, cidx[k]))
    if smoke:
        dev = dev[:200]
    vocab_counts = Counter(ch for t, _ in train for ch in _clean(t))
    vocab = {ch: i + 2 for i, (ch, n) in enumerate(vocab_counts.most_common()) if n >= (1 if smoke else 5)}
    return train, dev, classes, vocab


def evaluate(clf: MeterClassifier, data: list[tuple[str, int]], batch: int = 512, drop_marks: bool = False) -> float:
    correct = 0
    clf.model.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch):
            chunk = data[i : i + batch]
            pred = clf.model(clf.encode([t for t, _ in chunk], drop_marks)).argmax(-1).cpu().tolist()
            correct += sum(int(p == y) for p, (_, y) in zip(pred, chunk))
    return correct / max(1, len(data))


def train(a) -> int:
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    tr, dev, classes, vocab = build_dataset(a.roots, a.holdout, a.per_class_cap, a.min_class, a.seed, a.smoke)
    print(f"train {len(tr)}  dev {len(dev)}  classes {len(classes)}  vocab {len(vocab)}  ({time.time() - t0:.0f}s)", flush=True)
    cfg = {"d": a.d, "layers": a.layers, "heads": a.heads, "ff": a.ff, "dropout": a.dropout}
    model = MeterNet(len(vocab) + 2, len(classes), **cfg)
    clf = MeterClassifier(model, vocab, classes, a.device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr) / a.batch)
    total = 3 if a.smoke else steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(1, total), pct_start=0.05)
    dev_eval = dev if a.smoke else random.Random(a.seed).sample(dev, min(len(dev), a.dev_size))
    best, step = -1.0, 0
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(1 if a.smoke else a.epochs):
        random.Random(a.seed + epoch).shuffle(tr)
        clf.model.train()
        for i in range(0, len(tr), a.batch):
            chunk = tr[i : i + a.batch]
            drop = random.random() < 0.5  # learn to read unvowelised text too
            ids = clf.encode([t for t, _ in chunk], drop_marks=drop)
            y = torch.tensor([c for _, c in chunk], device=a.device)
            loss = loss_fn(clf.model(ids), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0 or a.smoke:
                print(f"epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if step >= total:
                break
        acc = evaluate(clf, dev_eval)
        acc_plain = evaluate(clf, dev_eval, drop_marks=True)
        print(f"epoch {epoch + 1}: dev hemistich acc {acc:.4f} (without harakat {acc_plain:.4f})", flush=True)
        if acc > best:
            best = acc
            clf.save(a.out, meta={"dev_acc": acc, "dev_acc_plain": acc_plain, "epoch": epoch + 1,
                                  "train_size": len(tr), "seed": a.seed})
        if step >= total:
            break
    print(f"best dev acc {best:.4f} -> {a.out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="train the neural metre classifier")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--roots", nargs="+", default=["data/raw/ganjoor"])
    t.add_argument("--out", default="out/aruz/classifier.pt")
    t.add_argument("--holdout", type=float, default=0.2)
    t.add_argument("--per-class-cap", type=int, default=25000)
    t.add_argument("--min-class", type=int, default=50)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--d", type=int, default=128)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--ff", type=int, default=384)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--dev-size", type=int, default=20000)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--device", default="cpu")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    if a.smoke and a.out == "out/aruz/classifier.pt":
        a.out = "out/smoke/aruz_classifier.pt"
    return train(a)


if __name__ == "__main__":
    raise SystemExit(main())

