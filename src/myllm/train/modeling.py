"""Model/tokenizer loading shared by every trainer.

Config block (``model:``)::

    model:
      name: Qwen/Qwen3-4B          # or tiny://qwen3 for offline smoke runs
      backend: hf                  # hf | unsloth   (unsloth: faster, less VRAM; GPU only)
      load_in_4bit: true           # QLoRA (NF4 + double quant); ignored without CUDA
      dtype: bfloat16
      attn_implementation: sdpa    # sdpa | flash_attention_2 | eager
      gradient_checkpointing: true
    lora:
      r: 16
      alpha: 32
      dropout: 0.05
      target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
      use_rslora: false
      use_dora: false
"""

from __future__ import annotations

from typing import Any

from myllm.config import get
from myllm.tiny_models import resolve_model_name

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _torch_dtype(name: str | None):
    import torch

    if name in (None, "auto"):
        return "auto"
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16,
            "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}[name]


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        return False


def bf16_supported() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:  # pragma: no cover
        return False


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    name = resolve_model_name(get(cfg, "model.tokenizer") or get(cfg, "model.name"))
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=get(cfg, "model.trust_remote_code", False))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if get(cfg, "model.chat_template_path"):
        with open(get(cfg, "model.chat_template_path"), encoding="utf-8") as fh:
            tok.chat_template = fh.read()
    return tok


def quantization_config(cfg: dict):
    """4-bit NF4 config for QLoRA (roadmap §3.1-c, §6.2), or None when not applicable."""
    if not get(cfg, "model.load_in_4bit", False) or not cuda_available():
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def model_init_kwargs(cfg: dict) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dtype": _torch_dtype(get(cfg, "model.dtype", "auto")) if cuda_available() else _torch_dtype("float32"),
        "trust_remote_code": get(cfg, "model.trust_remote_code", False),
    }
    attn = get(cfg, "model.attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn
    q = quantization_config(cfg)
    if q is not None:
        kwargs["quantization_config"] = q
        kwargs["device_map"] = {"": 0}
    return kwargs


def load_model(cfg: dict, name_key: str = "model.name"):
    """Load a causal LM honouring the profile (4-bit on GPU, fp32 on CPU)."""
    backend = get(cfg, "model.backend", "hf")
    name = resolve_model_name(get(cfg, name_key))
    if backend == "unsloth":
        return _load_unsloth(cfg, name)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, **model_init_kwargs(cfg))
    if get(cfg, "model.load_in_4bit", False) and cuda_available():
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=get(cfg, "model.gradient_checkpointing", True)
        )
    return model


def _load_unsloth(cfg: dict, name: str):  # pragma: no cover - GPU only
    """Unsloth path. Install it in a separate GPU env: ``pip install unsloth``.

    Unsloth must be imported before transformers in the *process*; run trainers as
    ``python -m myllm.train.sft ... --set model.backend=unsloth`` in a fresh interpreter.
    """
    from unsloth import FastLanguageModel  # type: ignore

    model, _tok = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=get(cfg, "train.max_length", 2048),
        load_in_4bit=get(cfg, "model.load_in_4bit", True),
        dtype=None,
    )
    lora = get(cfg, "lora", {}) or {}
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora.get("target_modules", DEFAULT_TARGETS),
        use_gradient_checkpointing="unsloth",
        use_rslora=lora.get("use_rslora", False),
        random_state=get(cfg, "seed", 42),
    )
    model._myllm_peft_applied = True
    return model


def lora_config(cfg: dict):
    """PEFT LoRA config, or None for full fine-tuning (``lora.enabled: false``)."""
    lora = get(cfg, "lora", {}) or {}
    if lora.get("enabled", True) is False:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 2 * lora.get("r", 16)),
        lora_dropout=lora.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get("target_modules", DEFAULT_TARGETS),
        use_rslora=lora.get("use_rslora", False),
        use_dora=lora.get("use_dora", False),
    )


def peft_config_for(model, cfg: dict):
    """Return the PEFT config to hand to a TRL trainer (None if Unsloth already applied LoRA)."""
    if getattr(model, "_myllm_peft_applied", False):
        return None
    return lora_config(cfg)


def training_kwargs(cfg: dict, smoke: bool = False) -> dict[str, Any]:
    """Common ``TrainingArguments`` fields from the ``train:`` block + profile."""
    t = dict(get(cfg, "train", {}) or {})
    on_gpu = cuda_available()
    kwargs: dict[str, Any] = {
        "output_dir": get(cfg, "output_dir", "out/run"),
        "num_train_epochs": t.get("epochs", 1),
        "per_device_train_batch_size": t.get("batch_size", 1),
        "gradient_accumulation_steps": t.get("grad_accum", 1),
        "learning_rate": float(t.get("learning_rate", 2e-4)),
        "lr_scheduler_type": t.get("lr_scheduler", "cosine"),
        "warmup_steps": t.get("warmup_steps", 0),
        "weight_decay": t.get("weight_decay", 0.0),
        "max_grad_norm": t.get("max_grad_norm", 1.0),
        "logging_steps": t.get("logging_steps", 10),
        "save_strategy": t.get("save_strategy", "epoch"),
        "save_total_limit": t.get("save_total_limit", 2),
        "seed": get(cfg, "seed", 42),
        "data_seed": get(cfg, "seed", 42),
        "bf16": on_gpu and bf16_supported(),
        "fp16": on_gpu and not bf16_supported(),
        "gradient_checkpointing": on_gpu and get(cfg, "model.gradient_checkpointing", True),
        "report_to": t.get("report_to", "none"),
        "optim": t.get("optim", "paged_adamw_8bit" if on_gpu and get(cfg, "model.load_in_4bit") else "adamw_torch"),
        "remove_unused_columns": t.get("remove_unused_columns", True),
    }
    if "warmup_ratio" in t:  # converted to steps by transformers>=5 when given as float warmup_steps
        kwargs["warmup_steps"] = float(t["warmup_ratio"])
    if t.get("max_steps"):
        kwargs["max_steps"] = t["max_steps"]
    if smoke:
        kwargs.update(max_steps=t.get("smoke_steps", 10), logging_steps=1, save_strategy="no", report_to="none")
    return kwargs
