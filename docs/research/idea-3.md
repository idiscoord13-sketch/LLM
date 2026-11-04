# ایده‌ی ۳ — Replay-Anchored Continual LoRA `[فرضیه]`

**مسئله:** آموزش هفتگی → فراموشی و drift.
**روش (`flywheel/continual.py`):** (۱) بافر replay = داده‌ی هفته‌های قبل + «لنگرها» (پاسخ‌های درست مدل مستقر روی golden)؛
(۲) مدل مستقر = reference در DPO/KTO (β = جریمه‌ی KL به آن)؛ (۳) هر N هفته ادغام TIES آداپترها؛ (۴) regression gate.

**آزمایش:** baseline = fine-tune متوالی ساده؛ متریک = کیفیت تازه **و** حفظ قدیمی (golden کامل + benchmark عمومی)؛
ablation = بدون replay / بدون لنگر / بدون merge. **ادبیات:** EWC، experience replay، model soups/merging موجودند؛ سهم ما مهندسیِ ترکیب.
