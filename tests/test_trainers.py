"""End-to-end smoke runs of every trainer on the tiny offline Qwen3 (a few seconds each on CPU).

These are the roadmap's "smoke-test before a long run" rule turned into CI: if a TRL/transformers
upgrade breaks an argument name or the data format, these fail before you rent a GPU.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("trl")
pytestmark = pytest.mark.slow


def _run(module, config, tmp_path, *extra):
    import importlib

    mod = importlib.import_module(f"myllm.train.{module}")
    out = tmp_path / module
    rc = mod.main(["--config", config, "--smoke", "--set", f"output_dir={out}", *extra])
    assert rc == 0
    assert (out / "adapter_config.json").exists() or (out / "config.json").exists()
    meta = (out / "run_config.yaml").read_text(encoding="utf-8")
    assert "git_commit" in meta and "versions" in meta  # reproducibility record
    return out


def test_sft(tmp_path):
    _run("sft", "configs/smoke/sft.yaml", tmp_path)


def test_sft_weighted(tmp_path):
    rows = [json.loads(x) for x in Path("data/samples/sft_fa.jsonl").read_text(encoding="utf-8").splitlines()]
    for i, r in enumerate(rows):
        r["weight"] = 0.2 + (i % 3) * 0.4  # verifier weights (idea #7) -> WeightedSamplerMixin
    data = tmp_path / "weighted.jsonl"
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    _run("sft", "configs/smoke/sft.yaml", tmp_path, "--set", f"data.train_files=[{data}]")


def test_dpo(tmp_path):
    _run("dpo", "configs/smoke/dpo.yaml", tmp_path)


def test_simpo(tmp_path):
    _run("dpo", "configs/smoke/simpo.yaml", tmp_path)


def test_kto_confidence_gated(tmp_path):
    rows = [json.loads(x) for x in Path("data/samples/kto_fa.jsonl").read_text(encoding="utf-8").splitlines()]
    for i, r in enumerate(rows):
        r["confidence"] = [0.9, 0.6, 0.2][i % 3]
    data = tmp_path / "kto.jsonl"
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = _run("kto", "configs/smoke/kto.yaml", tmp_path, "--set", f"data.train_files=[{data}]",
               "--set", "kto.min_confidence=0.3")
    assert (out / "review_queue.jsonl").exists()


def test_gkd(tmp_path):
    _run("gkd", "configs/smoke/gkd.yaml", tmp_path)


def test_grpo(tmp_path):
    _run("grpo", "configs/smoke/grpo.yaml", tmp_path)


def test_adapter_merge_and_eval(tmp_path):
    from myllm.eval import run_eval
    from myllm.train import merge

    sft = _run("sft", "configs/smoke/sft.yaml", tmp_path)
    merged = tmp_path / "merged"
    assert merge.main(["lora", "--base", "tiny://qwen3", "--adapter", str(sft), "--out", str(merged)]) == 0
    out = tmp_path / "eval"
    assert run_eval.main(["--config", "configs/smoke/eval.yaml", "--smoke", "--set", f"output_dir={out}",
                          "--set", f"model.model={merged}"]) == 0
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["n_items"] == 12 and "per_category" in res
