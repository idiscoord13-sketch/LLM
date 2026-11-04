# experiments/

- `registry.jsonl` — append-only log of every research run (`myllm.research.registry`). Never delete rows: negative results are results.
- `reports/<name>.md` — ablation reports with mean ± std and bootstrap CIs (`myllm.research.ablate`).
- `runs/` — per-run outputs (git-ignored; checkpoints are large).

Smoke rehearsals (`configs/smoke/ablation.yaml`) write to `out/smoke/` instead, so they never pollute this registry.
