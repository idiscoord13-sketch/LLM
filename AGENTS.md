# AGENTS.md — راهنمای پروژه برای agentها (همان محتوای CLAUDE.md)

## هدف پروژه
ساخت یک دستیار فارسی/انگلیسی + برنامه‌نویسی با post-training روی base باز (Qwen3، Apache-2.0)، یک GPT کوچک از صفر برای
یادگیری، و یک data flywheel هفتگی امن. سند مرجع: `docs/ROADMAP.md`. هدف کیفی: `configs/targets.yaml`.

## قواعد مطلق (نقض نکن)
- Python ≥ 3.11، مدیریت بسته با `uv`؛ نسخه‌ها در `uv.lock` pin هستند. Unsloth و vLLM در محیط جداگانه نصب می‌شوند.
- هرگز chat template را دستی نساز؛ همیشه `tokenizer.apply_chat_template()` (`src/myllm/data/chat_format.py`).
- loss فقط روی توکن‌های پاسخ assistant — `tests/test_chat_format.py` این را چک می‌کند؛ اگر آن را تغییر دادی تست را هم به‌روز کن.
- قبل از هر آموزش طولانی smoke-test بزن: `--smoke` (۱۰۰ نمونه، max_steps=10).
- seed همه‌جا ۴۲ (مگر در ablation)؛ هر run فایل `run_config.yaml` (config + git hash + نسخه‌ها) ذخیره می‌کند.
- داده‌ی خام کاربر، PII، `.env`، checkpoint و `data/raw|processed|flywheel` را هرگز commit نکن (`.gitignore`).
- هیچ API/آرگومان کتابخانه‌ای را حدس نزن؛ با نسخه‌ی نصب‌شده verify کن (`.venv/lib/python*/site-packages/trl/...`).
- روی benchmark آموزش نده؛ decontamination (`myllm.data.decontam`) در پایپ‌لاین اجباری است.
- هیچ ادعای «بهتر شد» بدون خروجی `myllm.eval.run_eval` + `myllm.flywheel.gate` (و برای پژوهش: `myllm.research.ablate` با چند seed).
- لایسنس: پیش‌فرض `commercial_safe: true`؛ داده/مدل NC را بدون تأیید صریح کاربر وارد نکن.

## دستورهای مهم
- نصب: `uv sync --extra train --extra serve --extra dev` (روی GPU: `--extra qlora` هم)
- تست: `uv run pytest -q` (تست‌های trainer با مارک `slow`؛ فقط سریع‌ها: `uv run pytest -q -m "not slow"`)
- پاک‌سازی داده: `uv run python -m myllm.data.pipeline --config configs/clean.yaml`
- smoke SFT: `uv run python -m myllm.train.sft --config configs/sft.yaml --profile local_4060 --smoke`
- آموزش آفلاین کامل (CPU): `uv run python -m myllm.train.<sft|dpo|kto|gkd|grpo> --config configs/smoke/<name>.yaml --smoke`
- eval: `uv run python -m myllm.eval.run_eval --config configs/eval.yaml`
- gate: `uv run python -m myllm.flywheel.gate --new <results.json> --baseline <results.json>`
- flywheel: `uv run python scripts/weekly_flywheel.py --config configs/flywheel.yaml --dry-run`
- ablation: `uv run python -m myllm.research.ablate --config configs/research/<idea>.yaml`
- عروض‌یار: `uv run python -m myllm.aruz.scan "<مصراع ۱>" "<مصراع ۲>"`، ارزیابی `uv run python scripts/eval_aruz.py --couplets`
  (داده‌ی گنجور در `data/raw/ganjoor` محلی است و commit نمی‌شود؛ لایسنسش روشن نیست)
- **کار بعدی برنامه‌ریزی‌شده:** به‌روزرسانی بزرگ عروض‌یار (دقت نزدیک به ۱۰۰٪ + گسترش محصول) — `docs/product/aruz_big_update.md`

## معماری کد
- `configs/profiles/*.yaml` = سخت‌افزار (مدل، batch، طول، ۴-بیت)؛ `configs/*.yaml` = recipe. ترتیب: recipe ← profile ← `--set`.
- `tiny://qwen3` = مدل Qwen3 ریز آفلاین برای smoke/تست (`src/myllm/tiny_models.py`).
- وزن هر نمونه (ستون `weight`) با `WeightedSamplerMixin` در `src/myllm/train/common.py` اعمال می‌شود (ایده‌های ۱، ۲، ۷).

## سبک
- تغییر کوچک، commit مکرر، پیام commit روشن. کد و کامنت انگلیسی، مستندات فارسی.
- در نقاط حساس (آموزش گران، حذف داده، انتشار مدل، تغییر لایسنس) قبل از اقدام از کاربر تأیید بگیر.
