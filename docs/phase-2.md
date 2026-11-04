# فاز ۲ — داده، تقطیر (distillation) و SFT

**هدف:** تبدیل base به دستیار فارسی+کد. اهرم اصلی کیفیت در مدل کوچک: **داده‌ی تمیز + تقطیر از teacher بزرگ با verifier**.

## ۱) پاک‌سازی (سند §۴.۲ — به همین ترتیب)
normalize فارسی → langID → dedup دقیق + MinHash → فیلتر کیفیت → حذف PII (موبایل/تلفن، ایمیل، کد ملی با checksum، کارت با Luhn، شبا) → decontamination
```bash
uv run python scripts/download_data.py --only farsinstruct tulu3_sft
uv run python -m myllm.data.pipeline --config configs/clean.yaml        # آمار قبل/بعد چاپ می‌شود
```
لایسنس: `commercial_safe: true` ردیف‌های NC را حذف می‌کند (`src/myllm/data/licenses.py`). کیفیت پیشرفته: برچسب ۰–۵ با LLM-judge
(`llm_quality_prompt`) روی چند هزار سند → `HashedQualityClassifier` → فیلتر میلیون‌ها سند (سبک FineWeb-Edu).

## ۲) تقطیر offline + rejection sampling (روی GPU اجاره‌ای)
```bash
# teacher بزرگ open-weight با vLLM روی کلاود (مثلاً Qwen3-235B-A22B یا Qwen3-32B)
uv run python -m myllm.distill.teacher_gen  --config configs/distill.yaml   # k پاسخ برای هر prompt
uv run python -m myllm.distill.reject_sample --config configs/distill.yaml  # verifier: تست کد / judge + self-consistency
```
خروجی: `sft_distilled.jsonl` (با `weight` در حالت `weighted` = **ایده‌ی ۷**) و `pref_distilled.jsonl` (جفت best/worst = داده‌ی DPO مجانی).
`thinking_mode` = **ایده‌ی ۶** (تفکر انگلیسی/پاسخ فارسی در برابر تفکر فارسی).

## ۳) SFT با QLoRA
```bash
uv run python -m myllm.train.sft --config configs/sft.yaml --profile local_4060 --smoke   # همیشه اول
uv run python -m myllm.train.sft --config configs/sft.yaml --profile local_4060
```
- ماسک loss مستقل از نسخه‌ی TRL ساخته می‌شود (`chat_format.py`)؛ برای Qwen3 که تاریخچه‌ی `<think>` را بازنویسی می‌کند fallback دارد.
- Unsloth (سریع‌تر، VRAM کمتر): در محیط جدا `pip install unsloth` و `--set model.backend=unsloth`.
- 4060 با 8GB: student 4B با QLoRA، `max_length` 2048، batch 1 × grad-accum 16. 8B فقط تجربی (Unsloth + seq 1024).

## معیار موفقیت
win-rate ≥ ۶۰٪ در برابر base روی golden set (`run_eval` با baseline + judge)، `code_pass@1` بالاتر، افت دانش عمومی < چند درصد.

## خطاهای رایج
template اشتباه، loss روی prompt، epoch زیاد (۱–۳ کافی است)، فراموشی decontam، داده‌ی مصنوعی از مدل‌هایی که ToS اجازه نمی‌دهد.
