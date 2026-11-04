"""myllm — a Persian/English assistant + coding LLM toolkit.

Sub-packages map one-to-one to the roadmap phases:
    data/      Phase 2 data pipeline (normalize, langid, dedup, quality, PII, decontam, chat format)
    tinygpt/   Phase 1 GPT-from-scratch lab (BPE, RoPE/RMSNorm/SwiGLU/GQA, pretrain, SFT)
    distill/   teacher generation + verifier-based rejection sampling
    train/     SFT / DPO / SimPO / KTO / GKD / GRPO / merging on open base models
    eval/      golden-set generation, pairwise LLM judge, code execution, benchmarks
    serve/     OpenAI-compatible gateway (auth, rate limit, consent logging, canary, test-time compute)
    flywheel/  weekly batch improvement loop with regression gate and rollback
    research/  experiment registry, ablations, statistics, original-idea implementations
"""

__version__ = "0.1.0"
