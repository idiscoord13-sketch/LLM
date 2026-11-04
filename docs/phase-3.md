# فاز ۳ — هم‌راستاسازی، استدلال و ادغام

| روش | فایل | کِی | سخت‌افزار |
|---|---|---|---|
| DPO / IPO / robust / APO | `train/dpo.py` (`preference.algorithm`) | پیش‌فرض بعد از SFT | 4060 (QLoRA) |
| SimPO (بدون reference) | `train/dpo.py` با `algorithm: simpo` | حافظه‌ی کم | 4060 |
| KTO (👍/👎) + **ایده‌ی ۲** | `train/kto.py` | فیدبک کاربران | 4060 |
| On-policy distillation (GKD) | `train/gkd.py` (TRL `DistillationTrainer`) | قوی‌ترین اهرم برای student کوچک | کلاود (teacher کنار student) |
| GRPO با پاداش «تست‌ها پاس شد» | `train/grpo.py` + `train/rewards.py` | کد/ریاضی، بعد از SFT+DPO | کلاود (vLLM) |
| ادغام TIES/DARE/SLERP/LoRA | `train/merge.py` | ترکیب student فارسی‌قوی و کدقوی | CPU/RAM |

```bash
uv run python -m myllm.train.dpo  --config configs/dpo.yaml  --profile local_4060 --smoke
uv run python -m myllm.train.kto  --config configs/kto.yaml  --profile local_4060
uv run python -m myllm.train.gkd  --config configs/gkd.yaml  --profile cloud_80gb
uv run python -m myllm.train.grpo --config configs/grpo.yaml --profile cloud_24gb
uv run python -m myllm.train.merge lora --base Qwen/Qwen3-4B-Instruct-2507 --adapter out/dpo --out out/dpo-merged
uv run python -m myllm.train.merge ties --base Qwen/Qwen3-4B-Instruct-2507 --models out/fa-merged out/code-merged --out out/merged
```

## نکته‌ها
- lr ترجیحی ~۵e-۶ (حدود ۴۰ برابر کمتر از SFT). اگر DPO روی داده‌ی کم overfit شد → `ipo` یا `beta` بالاتر یا `label_smoothing`.
- GKD: teacher و student باید **tokenizer یکسان** داشته باشند (یک خانواده). روی 4060 به‌جایش تقطیر offline (فاز ۲) را بزن.
- GRPO: `loss_type: dapo` + clip-higher (`epsilon_high`)؛ `beta: 0` یعنی بدون مدل مرجع (VRAM کمتر). پاداش‌ها وزن‌دار و قابل‌تأییدند
  (`code_tests`، `code_format`، `persian_prose`، `length_penalty` ضد reward hacking با طول).
- هر مرحله را با `run_eval` در برابر مرحله‌ی قبل بسنج؛ اگر بدتر شد، نگهش ندار.
