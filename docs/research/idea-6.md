# ایده‌ی ۶ — Cross-Lingual Thinking Bridge (تکرار + گسترش) `[فرضیه]`

**فرضیه:** مدل کوچک در انگلیسی بهتر استدلال می‌کند؛ «تفکر انگلیسی داخل `<think>`، پاسخ فارسی» دقت را بالا می‌برد بدون افت روانی فارسی.

**ادبیات — صادقانه: این ایده قبلاً منتشر شده.** Tran و همکاران «English-Pivoted CoT Training» را برای ریاضی به زبان ایرلندی
پیشنهاد کردند (arXiv 2504.02890، AAAI) و تا ۲۸٪ بهبود در برابر CoT کاملاً به زبان مقصد گزارش کردند. پس این‌جا **تازگی ادعا نمی‌کنیم**.
**سهم ما:** تکرار روی **فارسی**، گسترش از ریاضی به **کد و چت عمومی**، و سنجیدن هزینه‌ی روانی فارسی (judge + `persian_prose_reward`).

**پیاده‌سازی:** `distill/teacher_gen.py` → `thinking_mode: no_think | fa_think | en_think_fa_answer` (prompt سیستمی teacher)،
`distill.keep_think: true` تا تفکر در داده‌ی SFT بماند.
**آزمایش:** سه بازوی هم‌حجم داده × ۳ seed؛ متریک: `must_include_acc` روی `reasoning`، `code_pass@1`، win-rate روانی فارسی.
