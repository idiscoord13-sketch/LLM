import math

import torch

from myllm.tinygpt.bpe import ByteBPE
from myllm.tinygpt.mixing import MixingBandit
from myllm.tinygpt.model import GPT, GPTConfig, apply_rope, rope_frequencies
from myllm.tinygpt.pretrain import lr_at

CFG = GPTConfig(vocab_size=97, n_layer=2, n_head=4, n_kv_head=2, d_model=32, max_seq_len=32)


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    m = GPT(CFG)
    x = torch.randint(0, 97, (3, 10))
    logits, loss, caches = m(x, x)
    assert logits.shape == (3, 10, 97)
    assert math.isclose(loss.item(), math.log(97), rel_tol=0.15)  # ~uniform at init
    assert len(caches) == 2 and caches[0][0].shape == (3, 2, 10, 8)  # GQA: 2 kv heads


def test_causality_future_tokens_do_not_change_past():
    torch.manual_seed(0)
    m = GPT(CFG).eval()
    x = torch.randint(0, 97, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 1) % 97
    with torch.no_grad():
        a, _, _ = m(x)
        b, _, _ = m(y)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-5)
    assert not torch.allclose(a[0, 8:], b[0, 8:])


def test_kv_cache_matches_full_recompute():
    torch.manual_seed(0)
    m = GPT(CFG).eval()
    x = torch.randint(0, 97, (1, 6))
    out = m.generate(x, max_new_tokens=8, temperature=0)  # greedy, uses the KV cache
    seq = x
    with torch.no_grad():
        for _ in range(8):  # no cache: recompute the whole prefix each step
            logits, _, _ = m(seq)
            seq = torch.cat([seq, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, seq)


def test_rope_preserves_norm_and_is_relative():
    cos, sin = rope_frequencies(8, 16, 10000.0)
    q = torch.randn(1, 1, 16, 8)
    rq = apply_rope(q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), rq.norm(dim=-1), atol=1e-5)
    k = torch.randn(1, 1, 1, 8).expand(1, 1, 16, 8)
    qq = torch.randn(1, 1, 1, 8).expand(1, 1, 16, 8)
    rk, rqq = apply_rope(k, cos, sin), apply_rope(qq, cos, sin)
    # <R_m q, R_n k> depends only on m - n
    d1 = (rqq[0, 0, 5] * rk[0, 0, 3]).sum()
    d2 = (rqq[0, 0, 9] * rk[0, 0, 7]).sum()
    assert torch.allclose(d1, d2, atol=1e-4)


def test_overfits_a_single_batch():
    torch.manual_seed(0)
    m = GPT(CFG)
    opt = m.configure_optimizer(lr=3e-3, weight_decay=0.0)
    x = torch.randint(0, 97, (4, 16))
    first = None
    for _ in range(80):
        _, loss, _ = m(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.3 * first


def test_bpe_roundtrip_persian_and_compression():
    texts = ["زبان فارسی زیباست و زبان فارسی کهن است.", "the cat sat on the mat", "می‌خواهم کتاب‌ها را بخوانم."] * 5
    bpe = ByteBPE().train(texts, vocab_size=320)
    for t in texts[:3]:
        ids = bpe.encode(t)
        assert bpe.decode(ids) == t
        assert len(ids) < len(t.encode("utf-8"))  # merges actually compress


def test_lr_schedules():
    cfg = {"train": {"learning_rate": 1.0, "min_lr": 0.1, "max_steps": 100, "warmup_steps": 10, "schedule": "wsd",
                     "decay_fraction": 0.2}}
    assert lr_at(0, cfg) == 0.1 and lr_at(9, cfg) == 1.0
    assert lr_at(50, cfg) == 1.0  # stable phase
    assert 0.1 <= lr_at(99, cfg) < 0.2  # decayed at the end
    cfg["train"]["schedule"] = "cosine"
    assert lr_at(55, cfg) < 1.0 and abs(lr_at(100, cfg) - 0.1) < 1e-9


def test_bandit_moves_toward_improving_domain_with_floor():
    b = MixingBandit(["fa", "en"], [0.5, 0.5], {"fa": 2.0, "en": 4.0}, eta=3.0, min_prob=0.1)
    b.update({"fa": 5.0, "en": 5.0}, {"fa": 1000, "en": 1000})
    probs = b.update({"fa": 4.0, "en": 4.99}, {"fa": 1000, "en": 1000})  # fa improves much faster
    assert probs[0] > 0.5 > probs[1] >= 0.1
    assert abs(sum(probs) - 1) < 1e-9


def test_bandit_fertility_metric_is_bits_per_char():
    b = MixingBandit(["fa"], [1.0], {"fa": 2.0}, fertility_aware=True)
    assert math.isclose(b.metric("fa", 2.0), 2.0 / 2.0 / math.log(2))
    b2 = MixingBandit(["fa"], [1.0], {"fa": 2.0}, fertility_aware=False)
    assert b2.metric("fa", 2.0) == 2.0
