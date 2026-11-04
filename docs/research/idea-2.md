# ایده‌ی ۲ — Confidence-Gated KTO `[فرضیه]`

**فرضیه:** 👍/👎 نویزی است؛ اگر هر برچسب وزن اطمینان بگیرد و برچسب‌های کم‌اطمینان به بازبینی انسانی بروند، KTO بهتر می‌شود.
**پیاده‌سازی:** `flywheel/signals.py` — اطمینان = پایداری کاربر (توافق با اکثریت، Beta-smoothed) × توافق سیگنال‌ها (👍 + regenerate =
متناقض) × توافق judge (اختیاری). `train/kto.py` — زیر `kto.min_confidence` → `review_queue.jsonl`؛ بقیه نمونه‌گیری متناسب با اطمینان.

**آزمایش:** baseline = KTO بی‌وزن (`confidence_weighting: false`, `min_confidence: 0`)؛ متریک = golden + نرخ sycophancy روی
`category: trap`؛ ablation = حذف هر جزء اطمینان. ریسک: تقویت سوگیری کاربرانِ «قابل‌اعتمادِ» هم‌فکر → تنوع را رصد کن.

**ادبیات:** یادگیری با برچسب نویزی و reweighting شناخته‌شده‌اند (و DPO مقاوم — `algorithm: robust` در `train/dpo.py`). ترکیب خاص با
KTO/flywheel کمتر مستند است؛ ادعای تازگی قطعی نمی‌کنم.
