# فاز ۴ — ارزیابی و استقرار

## ارزیابی (`myllm.eval.run_eval`)
- متریک‌ها: `code_pass@1` (تست واقعی اجرا می‌شود)، `must_include_acc`، `safety_refusal`، `over_refusal`، و
  `win_rate_vs_baseline` با **جابه‌جایی A/B** (ضد position bias)، فاصله‌ی اطمینان bootstrap و sign-test.
- قاضی را از خانواده‌ی **متفاوت** با مدل تحت آزمون انتخاب کن؛ روی ۵۰ نمونه با داوری خودت کالیبره کن (`calibration_agreement`).
- benchmark فارسی (Khayyam/PersianMMLU، ParsiNLU، FarsEval): با `lm-evaluation-harness` (`scripts/select_base.py --lm-eval-tasks ...`).
- **بعد از کوانتایز دوباره eval بزن.**

## سرو روی 4060 (تک‌کاربر)
```bash
uv run python -m myllm.train.merge lora --base <base> --adapter out/dpo --out out/final
python llama.cpp/convert_hf_to_gguf.py out/final --outfile out/myllm-f16.gguf
llama.cpp/build/bin/llama-quantize out/myllm-f16.gguf out/myllm-Q4_K_M.gguf Q4_K_M
ollama create myllm -f scripts/Modelfile.example
```
## سرو چندکاربره (Linux/WSL2 یا کلاود)
```bash
UPSTREAM_API_KEY=... scripts/serve_vllm.sh out/final        # vLLM فقط روی 127.0.0.1
```
## Gateway (تنها چیزی که به شبکه باز می‌شود)
```bash
GATEWAY_API_KEYS=<کلید-طولانی-تصادفی> uv run python -m myllm.serve.gateway --config configs/gateway.yaml --host 0.0.0.0
```
- auth اجباری (بدون کلید بالا نمی‌آید)، rate limit، مسیریابی stable/canary از registry، لاگ **فقط با consent** (`X-Consent: true`)
  و PII حذف‌شده، `POST /v1/feedback` برای 👍/👎/ویرایش، `POST /v1/forget` برای حق حذف.
- **test-time compute**: `{"tts": {"n": 4}}` در body → چند پاسخ، انتخاب بهترین (کدِ قابل اجرا / رأی اکثریت). برای مدل کوچک ارزان و مؤثر.
- Open WebUI را به `http://<host>:8080/v1` با همان کلید وصل کن.
