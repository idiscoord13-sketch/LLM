# myllm — دستیار فارسی + برنامه‌نویسی، از صفر تا data flywheel

این ریپو پیاده‌سازی کامل [نقشه‌ی راه](docs/ROADMAP.md) است: از یک GPT کوچکِ «از صفر» برای یادگیری عمیق،
تا post-training پیشرفته روی یک base باز (Qwen3)، ارزیابی دقیق، سرو امن، و چرخه‌ی بهبود هفتگی از چت کاربران.

**هدف کیفی (North Star — [`configs/targets.yaml`](configs/targets.yaml)):** یک مدل **کوچک ۱٫۷B تا ۴B** که روی
golden set فارسی، کد (با اجرای واقعی تست‌ها) و benchmark فارسی، **مدل رسمیِ هم‌اندازه را شکست بدهد** و فاصله‌اش با مدل دو برابر بزرگ‌تر را
ببندد — و این ادعا فقط با عدد، فاصله‌ی اطمینان و چند seed پذیرفته می‌شود.

> **وضعیت صادقانه:** همه‌ی کد روی CPU و بدون اینترنت با یک مدل Qwen3 ریزِ تصادفی تست شده (`pytest` سبز: داده، ماسک loss،
> SFT/DPO/SimPO/KTO/GKD/GRPO، ادغام، eval، gateway، flywheel). **آموزش واقعی هنوز اجرا نشده** — آن روی RTX 4060 تو
> (یا GPU اجاره‌ای) انجام می‌شود. هیچ عددی از کیفیت مدل در این ریپو ادعا نشده؛ اعداد را eval تو می‌سازد.

## شروع سریع (RTX 4060)

ویندوز: آموزش را داخل **WSL2 (Ubuntu)** انجام بده (vLLM و بعضی کتابخانه‌ها روی ویندوزِ خام کار نمی‌کنند). لینوکس: مستقیم.

```bash
git clone <this repo> && cd LLM
curl -LsSf https://astral.sh/uv/install.sh | sh      # مدیر بسته
uv sync --extra train --extra qlora --extra serve --extra dev
uv run pytest -q                                     # باید همه سبز باشد (حدود ۲ دقیقه روی CPU)
```

اولین حلقه‌ی کامل (همان «اگر فردا صبح فقط یک کار بکنی» در سند):

```bash
uv run python scripts/fertility.py                                         # کدام tokenizer فارسی را ارزان‌تر می‌شکند؟
uv run python -m myllm.train.sft --config configs/smoke/sft.yaml --smoke   # smoke آفلاین، چند ثانیه
uv run python -m myllm.train.sft --config configs/sft.yaml --profile local_4060 --smoke   # روی مدل واقعی، ۱۰ گام
```

## نقشه‌ی فازها

| فاز | چه می‌سازد | راهنما | دستور اصلی |
|---|---|---|---|
| ۰ | محیط، golden set، انتخاب base با اندازه‌گیری | [phase-0](docs/phase-0.md) | `scripts/fertility.py`، `scripts/select_base.py` |
| ۱ | GPT از صفر (BPE، RoPE، GQA، SwiGLU، RMSNorm) + آزمایشگاه ترکیب داده | [phase-1](docs/phase-1.md) | `python -m myllm.tinygpt.pretrain` |
| ۲ | پاک‌سازی داده، تقطیر از teacher، rejection sampling، SFT با QLoRA | [phase-2](docs/phase-2.md) | `myllm.data.pipeline` → `myllm.distill.*` → `myllm.train.sft` |
| ۳ | DPO/SimPO/KTO، on-policy distillation (GKD)، GRPO با پاداش تست کد، ادغام مدل | [phase-3](docs/phase-3.md) | `myllm.train.{dpo,kto,gkd,grpo,merge}` |
| ۴ | ارزیابی دقیق + سرو (vLLM/Ollama) پشت gateway امن | [phase-4](docs/phase-4.md) | `myllm.eval.run_eval`، `myllm.serve.gateway` |
| ۵ | data flywheel هفتگی: consent → PII → سیگنال → داده → آموزش → gate → canary → rollback | [phase-5](docs/phase-5.md) | `scripts/weekly_flywheel.py` |
| پژوهش | ۷ ایده‌ی اورجینال + چارچوب ablation با آمار | [research](docs/research/README.md) | `python -m myllm.research.ablate` |
| محصول | **«عروض‌یار»**: تقطیع و تشخیص وزن و قافیه‌ی شعر فارسی برای درس علوم و فنون (بدون GPU) | [product/aruz](docs/product/aruz.md) | `python -m myllm.aruz.scan`، `python -m myllm.aruz.bot` |

## ساختار

```
configs/        recipe ها (چه آموزش بدهیم) + profiles/ (روی چه سخت‌افزاری: cpu_smoke, local_4060, cloud_24gb, cloud_80gb)
data/eval/      golden_fa.jsonl — ۶۲ نمونه‌ی دست‌نویس (عمومی، نوشتن، استدلال، ۱۴ کد با تست، تله، ایمنی)
data/samples/   داده‌ی کوچک برای smoke-test
src/myllm/      data/ tinygpt/ distill/ train/ eval/ serve/ flywheel/ research/
scripts/        fertility, select_base, download_data, serve_vllm.sh, Modelfile.example, weekly_flywheel
tests/          pytest — CPU، آفلاین
docs/           ROADMAP.md (سند اصلی) + راهنمای فارسی هر فاز + ایده‌های پژوهشی
```

## قواعد طلایی (از سند، در کد اجباری شده‌اند)

1. **اول eval، بعد آموزش.** golden set و `gate.py` قبل از هر انتشار.
2. **chat template فقط با `apply_chat_template`** و **loss فقط روی پاسخ assistant** — هر دو تست دارند.
3. **smoke-test قبل از هر آموزش طولانی** (`--smoke`).
4. **PII قبل از هر نوشتن داده حذف می‌شود**؛ بدون consent هیچ چیزی ذخیره نمی‌شود.
5. **هیچ‌وقت آموزش آنلاین per-message**؛ فقط batch هفتگی + regression gate + canary + rollback.
6. **endpoint بدون auth ممنوع** — gateway بدون کلید API اصلاً بالا نمی‌آید.

## «تجاری یا شخصی» یعنی چه؟ (سؤال لایسنس)

اگر روزی بخواهی مدل را به دیگران **بفروشی** یا در یک **محصول پولی** استفاده کنی، «تجاری» است. بعضی مدل‌ها و دیتاست‌ها
(مثل Aya Expanse یا ترجمه‌های Alpaca) فقط اجازه‌ی استفاده‌ی **غیرتجاری** می‌دهند. پیش‌فرض این ریپو **امن برای تجاری** است:
base مدل Qwen (Apache-2.0) و پایپ‌لاین داده ردیف‌های غیرتجاری را حذف می‌کند (`commercial_safe: true`).
اگر مطمئنی همیشه شخصی/پژوهشی می‌ماند، می‌توانی `commercial_safe: false` بگذاری. (این مشاوره‌ی حقوقی نیست.)

## برای agentها

قواعد کار با Claude Code / سایر agentها در [CLAUDE.md](CLAUDE.md) (و همان محتوا در [AGENTS.md](AGENTS.md)).
