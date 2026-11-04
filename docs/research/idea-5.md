# ایده‌ی ۵ — Self-Consistency Reward Bootstrapping `[فرضیه]`

**فرضیه:** بدون reward model فارسی خوب، توافق k نمونه + judge کوچکِ کالیبره یک پاداش ارزان می‌سازد؛ promptهای پرابهام → انسان.
**پیاده‌سازی:** `research/self_consistency.py` (توافق روی جواب نهایی یا شباهت n-gram؛ `uncertainty` برای active learning)،
ترکیب در `distill/reject_sample.py` (`self_consistency: true`) و انتخاب best-of-n در `serve/tts.py`.
**آزمایش:** baseline = judge تنها؛ متریک = همبستگی با داوری انسانی (۱۰۰ نمونه) + کیفیت نهایی. **ریسک مهم:** روی سؤال‌های ذهنی/خلاق
توافق بی‌معناست و اکثریت می‌تواند غلط باشد. **ادبیات:** self-consistency (Wang 2022)، RLAIF و active learning موجودند.
