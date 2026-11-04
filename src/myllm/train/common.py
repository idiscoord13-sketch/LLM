"""Shared trainer plumbing: data loading, per-sample weights, saving."""

from __future__ import annotations

import logging
from pathlib import Path

from myllm.config import get, resolve_path
from myllm.utils import read_jsonl, save_run_config, set_seed

log = logging.getLogger("myllm.train")


def load_rows(cfg: dict, key: str = "data.train_files", smoke: bool = False) -> list[dict]:
    files = get(cfg, key) or []
    if isinstance(files, str):
        files = [files]
    rows: list[dict] = []
    for f in files:
        rows.extend(read_jsonl(resolve_path(f)))
    limit = 100 if smoke else get(cfg, "data.max_samples")
    if limit:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError(f"no training rows found in {files}")
    return rows


class WeightedSamplerMixin:
    """Per-example weights without touching TRL internals.

    If the train dataset has a ``weight`` column, examples are drawn with
    probability proportional to their weight (with replacement). In expectation
    this equals multiplying each example's loss by its normalised weight, which is
    exactly what ideas #1 (edit-distance weighted DPO), #2 (confidence-gated KTO)
    and #7 (verifier-weighted distillation) need.
    """

    sample_weights: list[float] | None = None

    def _get_train_sampler(self, train_dataset=None):  # noqa: D401 - transformers hook
        import torch

        weights = self.sample_weights
        if weights is None:
            return super()._get_train_sampler(train_dataset)  # type: ignore[misc]
        n = len(train_dataset if train_dataset is not None else self.train_dataset)  # type: ignore[attr-defined]
        if len(weights) != n:
            log.warning("weight column length %d != dataset length %d; ignoring weights", len(weights), n)
            return super()._get_train_sampler(train_dataset)  # type: ignore[misc]
        gen = torch.Generator().manual_seed(int(self.args.seed))  # type: ignore[attr-defined]
        return torch.utils.data.WeightedRandomSampler(weights, num_samples=n, replacement=True, generator=gen)


def extract_weights(rows: list[dict]) -> list[float] | None:
    if not any("weight" in r for r in rows):
        return None
    weights = [max(0.0, float(r.get("weight", 1.0))) for r in rows]
    if sum(weights) <= 0:
        raise ValueError("all sample weights are zero")
    if len(set(weights)) == 1:
        return None  # uniform weights == ordinary shuffling
    return weights


def weighted_trainer_cls(base_cls):
    return type(f"Weighted{base_cls.__name__}", (WeightedSamplerMixin, base_cls), {})


def prepare_run(cfg: dict) -> Path:
    set_seed(get(cfg, "seed", 42))
    out = Path(get(cfg, "output_dir", "out/run"))
    save_run_config(out, cfg)
    return out


def finish(trainer, tokenizer, cfg: dict, smoke: bool = False) -> Path:
    out = Path(get(cfg, "output_dir", "out/run"))
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    metrics = {k: v for k, v in (trainer.state.log_history[-1] if trainer.state.log_history else {}).items()}
    save_run_config(out, cfg, extra={"final_log": metrics, "smoke": smoke})
    log.info("saved to %s", out)
    return out


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
