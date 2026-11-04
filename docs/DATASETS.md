# دیتاست‌ها و لایسنس

فهرست کامل با شناسه‌ی Hugging Face، فیلدها و لایسنس: [`configs/data_sources.yaml`](../configs/data_sources.yaml).
دانلود: `uv run python scripts/download_data.py --only <names>`.

- ردیف‌هایی که `verify: true` دارند از ماشین سازنده قابل بررسی نبودند (دسترسی HF بسته بود) — شناسه و لایسنس را روی Hub تأیید کن.
- هر چیزی که از Stanford Alpaca ترجمه/مشتق شده **غیرتجاری** است (CC BY-NC).
- داده‌هایی که با GPT-4 و سرویس‌های بسته تولید شده‌اند ممکن است محدودیت ToS داشته باشند؛ برای داده‌ی مصنوعیِ خودت از teacherهای
  open-weight با لایسنس مجاز (Qwen، Apache-2.0) استفاده کن.
- benchmarkهای فارسی (Khayyam/PersianMMLU، ParsiNLU، FarsEval) **فقط برای ارزیابی**‌اند؛ هرگز رویشان آموزش نده و در
  `configs/clean.yaml → decontam.eval_files` اضافه‌شان کن.
