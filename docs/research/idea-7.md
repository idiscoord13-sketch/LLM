# ایده‌ی ۷ — Verifier-Weighted Distillation `[فرضیه]`

**فرضیه:** در rejection sampling کلاسیک پاسخ‌های «نیمه‌درست» دور ریخته می‌شوند؛ اگر به‌جای فیلتر سخت، هر نمونه با امتیاز verifier
(کسر تست‌های پاس‌شده / نمره‌ی judge) وزن بگیرد، داده‌ی بیشتری با سیگنال کمتر-ولی-مفید داریم.
**پیاده‌سازی:** `distill/reject_sample.py` (`mode: weighted`، `top_m`) → ستون `weight` → `WeightedSamplerMixin` در SFT.
**آزمایش:** `mode: hard` در برابر `weighted` با compute برابر؛ متریک: `code_pass@1` + golden. ablation: آستانه‌های مختلف.
**ادبیات:** reward-weighted regression/ReST/RAFT شناخته‌شده‌اند؛ یک پیش‌چاپ ۲۰۲۶ (arXiv 2605.13501، Reward-Weighted On-Policy
Distillation) نسخه‌ی on-policy با KL متراکم teacher را بررسی کرده و گزارش داده بیشتر سود از سیگنال متراکم teacher است تا خود فیلتر —
یعنی بعد از این آزمایش، `train/gkd.py` (on-policy) را با وزن verifier ترکیب کن.
