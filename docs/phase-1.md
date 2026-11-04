# فاز ۱ — GPT از صفر + آزمایشگاه پژوهشی ارزان

**هدف:** فهم عمیق tokenizer → معماری → pre-training → SFT، با کدی که خط‌به‌خط قابل توضیح است. (سند §۵.۴)

## فایل‌ها
| فایل | چه چیزی یاد می‌دهد |
|---|---|
| `src/myllm/tinygpt/bpe.py` | BPE بایت‌سطح از صفر (~۸۰ خط) — همان الگوریتم tokenizer مدل‌های واقعی |
| `src/myllm/tokenizer_utils.py` | همان BPE با کتابخانه‌ی Rust (`tokenizers`) + chat template ChatML |
| `src/myllm/tinygpt/model.py` | RMSNorm، RoPE، GQA، SwiGLU، QK-norm، KV cache — اجزای Llama/Qwen در ~۲۰۰ خط |
| `src/myllm/tinygpt/pretrain.py` | AdamW، warmup + cosine/WSD، grad accumulation، bf16، eval هر حوزه، checkpoint |
| `src/myllm/tinygpt/mixing.py` | **ایده‌ی ۴**: bandit ترکیب داده‌ی آگاه به fertility |
| `src/myllm/tinygpt/sft.py` | SFT با همان ماسک loss مدل‌های بزرگ |

## اجرا
```bash
# داده (روی سیستم خودت؛ نیاز به اینترنت/HF)
uv run python scripts/download_data.py --only naab fineweb_edu code --max-rows 300000
# آموزش ~31M پارامتر روی 4060 (چند ساعت؛ اول با --smoke)
uv run python -m myllm.tinygpt.pretrain --config configs/tiny_gpt.yaml --smoke
uv run python -m myllm.tinygpt.pretrain --config configs/tiny_gpt.yaml
uv run python -m myllm.tinygpt.sft --config configs/tiny_gpt.yaml
uv run python -m myllm.tinygpt.generate --ckpt out/tinygpt/ckpt_sft.pt --chat "سلام! خودت را معرفی کن."
```
محاسبه‌ی compute: C ≈ 6·N·D = 6 × 31M × 600M ≈ 1.1×10¹⁷ FLOPs. لاگ: `out/tinygpt/log.jsonl` (loss هر حوزه، lr، نسبت ترکیب، توکن/ثانیه).

## آزمایش‌های پیشنهادی (هرکدام چند ساعت روی 4060)
- `train.schedule: wsd` در برابر `cosine` (و ادامه‌ی آموزش از وسط WSD).
- `model.n_kv_head`: 8 (MHA) در برابر 2 (GQA) — کیفیت در برابر حافظه‌ی KV cache.
- **ایده‌ی ۴**: `uv run python -m myllm.research.ablate --config configs/research/idea4_mixing.yaml`.

## معیار موفقیت
loss به‌طور معنادار پایین بیاید، مدل متن فارسی (ولو ضعیف) تولید کند، و بتوانی هر خط `model.py` را توضیح بدهی.
انتظار کیفیت محصول از این مدل نداشته باش — هدف یادگیری و آزمایشگاه ارزان است.
