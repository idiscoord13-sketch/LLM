# فاز ۰ — آماده‌سازی، golden set، انتخاب base با اندازه‌گیری

**هدف:** محیط کار + معیار سنجش *قبل* از هر آموزش. (سند §۱۱، §۸)

## گام‌ها
1. نصب: `uv sync --extra train --extra qlora --extra serve --extra dev` و `uv run pytest -q`.
2. ورود به Hugging Face برای مدل‌ها/دیتاست‌های gated: `huggingface-cli login` (توکن در `.env`، هرگز commit نشود).
3. **golden set** را مال خودت کن: `data/eval/golden_fa.jsonl` با ۶۲ نمونه شروع می‌شود؛ تا ۲۰۰–۵۰۰ نمونه از سؤال‌های واقعیِ
   خودت/کاربرانت گسترش بده. فیلدها: `prompt`، `category`، و هرکدام لازم بود: `must_include` (حقایق لازم)، `tests` (کد)،
   `should_refuse` (ایمنی)، `reference` (راهنمای داور). **۲۰٪ را کنار بگذار و هرگز هنگام توسعه نگاه نکن** (test خصوصی).
4. **fertility** فارسی: `uv run python scripts/fertility.py` — توکن به‌ازای هر کلمه برای Qwen/Gemma/Llama/Mistral/Aya.
5. **انتخاب base با عدد**: هر کاندید را با vLLM یا Ollama بالا بیاور و:
   ```bash
   uv run python scripts/select_base.py --candidates Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Instruct-2507 \
       --endpoint Qwen/Qwen3-4B-Instruct-2507=http://localhost:8000/v1
   ```
   خروجی: `out/select_base/report.md`. نام نسخه‌ها سریع عوض می‌شوند — جدیدترین نسخه‌ی هر خانواده را روی Hub ببین.

## روی 4060 (8GB)
- اجرای مدل 4B برای انتخاب base: Ollama با GGUF Q4_K_M راحت جا می‌شود؛ 8B در Q4 هم (~۵GB) با context متوسط.
- اگر 4060 Ti با 16GB داری، در `configs/profiles/local_4060.yaml` مقدار `batch_size: 2` و `max_length: 4096` بگذار.

## معیار موفقیت
مدل آماده را لود کنی، روی golden set خروجی بگیری و `myllm.eval.run_eval` یک `results.json` بدهد (baseline تو).

## خطاهای رایج
پریدن سر آموزش بدون eval؛ نصب درهم نسخه‌ها (همیشه از `uv.lock`)؛ اعتماد به benchmark عمومی به‌جای golden set خودت.
