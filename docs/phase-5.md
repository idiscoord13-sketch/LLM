# فاز ۵ — Data Flywheel هفتگی

```
[چت] → (۱) لاگ با consent → (۲) حذف PII + نام مستعار → (۳) سیگنال (👍/👎، regenerate، ویرایش) + اطمینان →
(۴) فیلتر (prompt-injection، toxic، dedup، سقف سهم هر کاربر) → (۵) SFT/DPO/KTO + replay →
(۶) آموزش هفتگی (KTO/DPO) → (۷) eval + regression gate → (۸) canary → promote یا rollback
```
| مرحله | فایل |
|---|---|
| ۱–۲ | `flywheel/store.py` (+ gateway) |
| ۳ | `flywheel/signals.py` — اطمینان = پایداری کاربر × توافق سیگنال‌ها × (اختیاری) توافق judge (**ایده‌ی ۲**) |
| ۴ | `flywheel/filters.py` |
| ۵ | `flywheel/build_dataset.py` — **ایده‌ی ۱**: ویرایش کاربر = chosen، با باند فاصله‌ی ویرایش و وزن |
| ۶ | `train/kto.py` / `train/dpo.py` با مدل مستقر به‌عنوان reference (لنگر KL — **ایده‌ی ۳**، `flywheel/continual.py`) |
| ۷ | `flywheel/gate.py` (exit≠0 در صورت افت — مناسب CI) |
| ۸ | `flywheel/release.py` (registry، canary درصدی چسبنده، rollback فوری) |

```bash
uv run python scripts/weekly_flywheel.py --config configs/flywheel.yaml --import data/samples/flywheel_logs.jsonl --dry-run
uv run python scripts/weekly_flywheel.py --config configs/flywheel.yaml            # با GPU: آموزش + eval + gate + canary
uv run python -m myllm.flywheel.release status
uv run python -m myllm.flywheel.release rollback                                   # اگر متریک زنده افت کرد
```
## قواعد
- هیچ‌وقت آموزش online per-message. همیشه batch + gate + rollback.
- حداقل `min_rows_to_train` ردیف (پیش‌فرض ۲۰۰) قبل از آموزش؛ زیر آن فقط جمع کن.
- هر ۴ هفته adapterهای هفتگی را با TIES ادغام کن (`continual.consolidate_cmd`) به‌جای انباشت بی‌پایان.
- sycophancy را رصد کن: سؤال‌های «تله»ی golden set (`category: trap`) باید بعد از هر دور پاس بمانند.
