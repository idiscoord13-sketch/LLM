from myllm.clients import FunctionClient
from myllm.distill.reject_sample import run, score_candidates
from myllm.distill.teacher_gen import THINKING_SYSTEM, build_messages, generate
from myllm.research.self_consistency import agreement_scores, final_answer
from myllm.train.kto import balanced_class_weights, gate_by_confidence

ROW = {
    "id": "t1",
    "prompt": [{"role": "user", "content": "add"}],
    "tests": "assert add(1, 2) == 3\nassert add(0, 0) == 0",
    "candidates": [
        "```python\ndef add(a, b):\n    return a + b\n```",
        "```python\ndef add(a, b):\n    return a * b\n```",  # passes 1/2 (0*0 == 0)
        "```python\ndef add(a, b):\n    return a - b - 1\n```",
    ],
}


def test_verifier_scores():
    assert score_candidates(ROW) == [1.0, 0.5, 0.0]


def test_hard_vs_weighted_selection_and_pairs():
    sft, pairs = run([ROW], {"distill": {"mode": "hard", "threshold": 0.5}})
    assert len(sft) == 1 and sft[0]["weight"] == 1.0
    sft_w, _ = run([ROW], {"distill": {"mode": "weighted", "threshold": 0.4, "top_m": 2}})
    assert [r["weight"] for r in sft_w] == [1.0, 0.5]  # idea #7: partial credit with less pull
    assert len(pairs) == 1 and "a + b" in pairs[0]["chosen"][0]["content"]


def test_teacher_gen_with_function_client():
    client = FunctionClient(lambda msgs, n: [f"ans{i}:{msgs[0]['role']}" for i in range(n)], model="fake")
    out = generate([{"id": 1, "prompt": "سلام"}], client, k=3, mode="en_think_fa_answer", params={})
    assert out[0]["candidates"] == ["ans0:system", "ans1:system", "ans2:system"]
    assert build_messages([{"role": "user", "content": "x"}], "fa_think")[0]["content"] == THINKING_SYSTEM["fa_think"]


def test_self_consistency():
    scores, unc = agreement_scores(["جواب ۴۲", "42", "the answer is 42", "41"])
    assert scores[:3] == [0.75, 0.75, 0.75] and scores[3] == 0.25 and abs(unc - 0.25) < 1e-9
    assert final_answer("<think>1+1=2</think> نتیجه ۱۰٫۵") == "10.5" or final_answer("نتیجه 10.5") == "10.5"


def test_kto_confidence_gate():
    rows = [{"label": True, "confidence": 0.9}, {"label": False, "confidence": 0.1}, {"label": False, "confidence": 0.5}]
    keep, review = gate_by_confidence(rows, 0.3)
    assert len(keep) == 2 and len(review) == 1
    dw, uw = balanced_class_weights([{"label": True}, {"label": False}, {"label": False}])
    assert dw == 2.0 and uw == 1.0
