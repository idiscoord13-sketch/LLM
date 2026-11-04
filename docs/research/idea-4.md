# ایده‌ی ۴ — Fertility-Aware Adaptive Data Mixing `[فرضیه]`

**فرضیه:** نسبت ثابت فارسی/انگلیسی/کد بهینه نیست؛ bandit که بهبود را به‌ازای هر توکن مصرفی و بر حسب **bits-per-character**
(نه per-token) می‌سنجد، با همان compute فارسی بهتری می‌دهد.
**پیاده‌سازی:** `tinygpt/mixing.py` (exponentiated-gradient حول prior + کف اکتشاف `min_prob`).
**آزمایش (ارزان، روی 4060):** `configs/research/idea4_mixing.yaml` — fixed / bandit_per_token / bandit_fertility × ۳ seed.
متریک: loss اعتبارسنجی فارسی (و انگلیسی/کد برای کنترل فراموشی). **ادبیات:** DoReMi (Xie 2023) و online data mixing (Albalak 2023)
بسیار نزدیک‌اند؛ افزوده‌ی ما جمله‌ی fertility و تمرکز فارسی → تازگی محدود، ارزش عملی.
