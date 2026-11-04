# ترک پژوهشی — ایده‌ها، روش سنجش، و وضعیت ادبیات

هدف: «بهترین کیفیت در مدل کوچک» را **قابل اثبات** کنیم، نه شعار. هر ایده یک فرضیه است تا وقتی عدد بیاید.

## قواعد شواهد (`configs/targets.yaml` → `evidence_rules`)
1. baseline صریح + ablation (هر جزء را جدا خاموش کن).
2. حداقل ۳ seed آموزش؛ گزارش میانگین ± انحراف و **CI بوت‌استرپ** (`myllm.research.stats`).
3. golden set خصوصی (۲۰٪ دست‌نخورده) + benchmark عمومی **decontaminate‌شده**.
4. نتایج منفی هم ثبت می‌شوند (`experiments/registry.jsonl` فقط append).
5. قبل از ادعای «تازگی»، ادبیات را بگرد و در فایل ایده بنویس چه کسی قبلاً چه کرده.

## ابزار
```bash
uv run python -m myllm.research.ablate --config configs/research/idea4_mixing.yaml   # واقعی (4060)
uv run python -m myllm.research.ablate --config configs/smoke/ablation.yaml          # تمرین آفلاین
```
گزارش Markdown در `experiments/reports/<name>.md` (میانگین، std، Δ نسبت به baseline، CI).

## ایده‌ها
| # | ایده | کجا در کد | وضعیت تازگی (بعد از جست‌وجوی ادبیات، نوامبر ۲۰۲۶) |
|---|---|---|---|
| 1 | [Edit-Signal Preference Mining](idea-1.md) | `flywheel/build_dataset.py` | **کار مشابه منتشر شده**؛ افزوده‌ی ما: باند فاصله + وزن + حلقه‌ی فارسی |
| 2 | [Confidence-Gated KTO](idea-2.md) | `flywheel/signals.py`، `train/kto.py` | ترکیب خاص کم‌مستند؛ اجزا شناخته‌شده |
| 3 | [Replay-Anchored Continual LoRA](idea-3.md) | `flywheel/continual.py`، `train/merge.py` | مهندسیِ ترکیب روش‌های موجود |
| 4 | [Fertility-Aware Adaptive Mixing](idea-4.md) | `tinygpt/mixing.py` | نزدیک به DoReMi؛ افزوده: سنجه‌ی bits-per-char |
| 5 | [Self-Consistency Reward Bootstrapping](idea-5.md) | `research/self_consistency.py`، `distill/reject_sample.py` | ترکیب روش‌های موجود |
| 6 | [Cross-Lingual Thinking Bridge](idea-6.md) | `distill/teacher_gen.py` (`thinking_mode`) | **قبلاً منتشر شده** (English-Pivoted CoT) → تکرار + گسترش به فارسی/کد |
| 7 | [Verifier-Weighted Distillation](idea-7.md) | `distill/reject_sample.py` (`mode: weighted`) | کار نزدیک ۲۰۲۶ (reward-weighted OPD)؛ نسخه‌ی offline ما ساده‌تر |
| — | [Memory→Weights Promotion Ladder](idea-memory.md) | فقط طراحی | بعد از فاز ۵ |

> **صداقت:** تا امروز هیچ‌کدام از این ایده‌ها در این ریپو روی مدل واقعی آزموده نشده؛ کد و چارچوب سنجش آماده است.
> شانس واقعیِ «چیزی که کسی به آن نرسیده» بیشتر در **ترکیب** این‌ها روی **فارسی + کد در مقیاس کوچک** و در **داده‌ی واقعی flywheel تو** است
> (داده‌ای که هیچ آزمایشگاه دیگری ندارد) — نه در یک ترفند تکی.
