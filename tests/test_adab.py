"""Rule-based literary devices (myllm.adab.devices): precision rules found by manual review of Hafez."""

from __future__ import annotations

import pytest

from myllm.adab.devices import _jenas_type, find_devices, format_devices, tokenize


def _jenas(a: str, b: str) -> str | None:
    w = tokenize([f"{a} {b}"])
    t = _jenas_type(w[0], w[1])
    return t[0] if t else None


@pytest.mark.parametrize("a,b,kind", [
    ("گل", "دل", "جناس ناقص اختلافی"),
    ("باقی", "ساقی", "جناس ناقص اختلافی"),
    ("امان", "زمان", "جناس ناقص اختلافی"),  # initial ا is a glottal stop: one sound differs
    ("جام", "جم", "جناس ناقص افزایشی"),
    ("مه", "مهر", "جناس ناقص افزایشی"),
    ("باد", "باده", "جناس ناقص افزایشی"),  # final ه makes a new word here
    ("محرم", "حرم", "جناس ناقص افزایشی"),
    ("گُل", "گِل", "جناس ناقص حرکتی"),
])
def test_jenas_found(a, b, kind):
    assert _jenas(a, b) == kind


@pytest.mark.parametrize("a,b", [
    ("راه", "ره"), ("آینه", "آیینه"), ("آورید", "آرید"), ("نیک", "نیکو"),  # spelling variants
    ("بوی", "بویی"), ("دل", "دلم"), ("رفت", "رفته"), ("شکست", "شکسته"),  # grammatical endings
    ("فکنی", "فکندی"), ("بخور", "مخور"), ("برود", "نرود"), ("مبین", "بین"),  # one verb, different forms
    ("آب", "شب"), ("دلم", "دام"),  # more than one sound differs
])
def test_jenas_rejected(a, b):
    assert _jenas(a, b) is None


def test_izafe_kasra_is_not_harakat_jenas():
    words = tokenize(["به سرِ سبز", "از سَر بنه"])
    sar = [w for w in words if w.plain == "سر"]
    t = _jenas_type(sar[0], sar[1])
    assert t is None or not t[0].endswith("حرکتی")


def test_light_verbs_are_not_jenas():
    kinds = {d.kind for d in find_devices(["ناز کم کن که در این باغ", "آنچه نبود نشود"])}
    assert not any(k.startswith("جناس ناقص") for k in kinds)


def test_find_devices_on_classic_couplet():
    devices = find_devices(["هر آن که کنج قناعت به گنج دنیا داد", "فروخت یوسف مصری به کمترین ثمنی"])
    jenas = [d for d in devices if d.kind == "جناس ناقص اختلافی"]
    assert {w.plain for w in jenas[0].words} == {"کنج", "گنج"}
    text = format_devices(devices)
    assert "کنج" in text and "استعاره" in text  # explains what it does not detect


def test_antithesis_harmony_simile():
    kinds = {d.kind for d in find_devices(["روز و شب همچو شمع می‌سوزم", "پیش پروانه در میان بزم"])}
    assert "تضاد" in kinds
    assert "تشبیه" in kinds
    assert "تناسب (مراعات نظیر)" in kinds  # شمع، پروانه، بزم


def test_radif_is_not_repetition():
    devices = find_devices(["ساقیا برخیز و درده جام را غم مخور", "خاک بر سر کن غم ایام را غم مخور"])
    assert not any(d.kind == "تکرار" and d.words[0].plain in {"غم", "مخور"} for d in devices)


def test_bot_arayeh_command():
    from myllm.aruz.bot import AruzService

    s = AruzService(None)
    out = s.handle_text("/arayeh\nهر آن که کنج قناعت به گنج دنیا داد\nفروخت یوسف مصری به کمترین ثمنی")
    assert "جناس" in out and "کنج" in out
    assert "/arayeh" in s.handle_text("/arayeh")


# --- literary history question bank ------------------------------------------------------------------
def test_canon_integrity():
    from myllm.adab.history import CANON, ORDINALS, STYLES, authors_of, century

    assert len({a.name for a in CANON}) == len(CANON)
    for a in CANON:
        assert a.style is None or a.style in STYLES
        if a.death is not None:
            assert not a.modern and century(a.death) in ORDINALS
    shared = {w for w, owners in authors_of().items() if len(owners) > 1}
    assert shared == {"لیلی و مجنون"}  # نظامی and جامی; never asked as «اثر کیست؟»


def test_history_questions_are_well_formed():
    from myllm.adab.history import CANON, authors_of, build

    owners = authors_of()
    by_name = {a.name: a for a in CANON}
    bank = build(120, seed=3)
    assert len(bank) >= 100
    for q in bank:
        assert len(q.options) == 4 and len(set(q.options)) == 4
        assert q.options[q.answer] == q.answer_text
        if q.kind == "work_author":
            work = q.prompt.split("«")[1].split("»")[0]
            assert owners[work] == {q.answer_text}
        if q.kind == "not_by":
            author = by_name[q.prompt.removeprefix("کدام اثر از ").removesuffix(" نیست؟")]
            assert q.answer_text not in author.works
            assert all(o in author.works for i, o in enumerate(q.options) if i != q.answer)


def test_history_facts():
    import random

    from myllm.adab.history import CANON, q_century, q_title, q_work_author

    hafez = next(a for a in CANON if a.name == "حافظ")
    saadi = next(a for a in CANON if a.name == "سعدی")
    rng = random.Random(0)
    assert q_century(hafez, rng).answer_text == "قرن هشتم"
    assert q_title(hafez, rng).prompt == "لقب «لسان‌الغیب» از آنِ کیست؟"
    q = q_work_author(saadi, rng)
    assert q.answer_text == "سعدی" and ("بوستان" in q.prompt or "گلستان" in q.prompt)


def test_verse_filters(tmp_path):
    import json

    from myllm.adab.history import _persian, doubtful_prefixes

    assert _persian(["دلا بسوز که سوز تو کارها بکند", "نیاز نیم‌شبی دفع صد بلا بکند"])
    assert not _persian(["أَلا یا أَیُّهَا السّاقی أَدِرْ کَأْساً وَ ناوِلْها"])  # Arabic hemistich
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "khajoo.json").write_text(json.dumps(
        [{"title": "سام نامه - سراینده نامعلوم منسوب به خواجو", "url": "/khajoo/samname"},
         {"title": "دیوان اشعار", "url": "/khajoo/divan"}], ensure_ascii=False), encoding="utf-8")
    assert doubtful_prefixes("khajoo", tmp_path) == ["/khajoo/samname/"]


def test_bot_tarikh(tmp_path):
    import json
    import random
    from dataclasses import asdict

    from myllm.adab.history import CANON, q_century
    from myllm.aruz.bot import AruzService

    q = q_century(next(a for a in CANON if a.name == "حافظ"), random.Random(0))
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(asdict(q), ensure_ascii=False) + "\n", encoding="utf-8")
    s = AruzService(None, history_path=str(path))
    out = s.handle_text("/tarikh", chat_id=1)
    assert "حافظ" in out and "قرن هشتم" in out
    reply = s.handle_text("۱۲۳۴"[q.answer], chat_id=1)  # answers may come in Persian digits
    assert reply.startswith("✅") and "/tarikh" in reply


# --- adaptive tutor -----------------------------------------------------------------------------------
def _toy_bank():
    from myllm.aruz.exercises import Question

    bank = []
    for kind in ("meter", "century"):
        for k in range(6):
            bank.append(Question(kind, f"{kind} سؤال {k}", [], ["الف", "ب", "ج", "د"], 0, "الف"))
    return bank


def test_tutor_focuses_on_weak_skill_and_reviews_mistakes(tmp_path):
    from myllm.adab.tutor import Tutor, qkey

    bank = _toy_bank()
    t = Tutor(seed=1, path=tmp_path / "profiles.json", salt="s")
    for q in bank:  # meter always wrong, century always right
        t.record_answer("u", q, correct=q.kind == "century")
    picks = [t.pick("u", bank).kind for _ in range(40)]
    assert picks.count("meter") > picks.count("century")  # weak skill first (plus reviews of wrong ones)
    wrong = next(q for q in bank if q.kind == "meter")
    t2 = Tutor(seed=1, salt="s")
    t2.record_answer("v", wrong, correct=False)
    seen = [qkey(t2.pick("v", bank)) for _ in range(4)]
    assert qkey(wrong) in seen  # comes back after 3 questions
    # stored under a salted hash, never the raw user id
    raw = (tmp_path / "profiles.json").read_text(encoding="utf-8")
    assert '"u"' not in raw and Tutor(path=tmp_path / "profiles.json", salt="s").profile("u").stats["meter"] == [6, 0]


def test_tutor_report_card():
    from myllm.adab.tutor import Tutor
    from myllm.aruz.diagnose import Diagnosis, Mistake

    t = Tutor(salt="")
    assert "/tamrin" in t.report("x")
    for q in _toy_bank():
        t.record_answer("x", q, correct=q.kind == "century")
    d = Diagnosis("مصراع", "-u--", "--u-", None, [Mistake(2, "نی", "u", "-", "…", "noon_after_long")], correct=2)
    t.record_check("x", d)
    card = t.report("x")
    assert "تشخیص وزن" in card and "بیشترین نیاز به تمرین" in card and "«ن» پس از مصوت بلند" in card


def test_bot_karname():
    from myllm.aruz.bot import AruzService

    s = AruzService(None)
    assert "/tamrin" in s.handle_text("/karname", chat_id=5)


# --- printable exam -----------------------------------------------------------------------------------
def test_exam_versions_share_questions_and_keys_match():
    from myllm.adab.exam import build_exam
    from myllm.adab.history import build

    bank = build(80, seed=5)
    doc, sets = build_exam(bank, "history:12", versions=2, seed=9, title="آزمون <تست>", minutes=30)
    assert doc.startswith("<!doctype html>") and 'dir="rtl"' in doc and "&lt;تست&gt;" in doc
    a, b = sets

    def key(q):  # chronology questions share a prompt, so the options are part of the identity
        return q.prompt, tuple(q.verse), frozenset(q.options)

    assert {key(q) for q in a} == {key(q) for q in b}
    twins = {key(q): q for q in a}
    for q in b:  # same correct option text after shuffling
        twin = twins[key(q)]
        assert q.options[q.answer] == twin.options[twin.answer]
    assert doc.count("پاسخ‌نامه") == 2 and "۳۰ دقیقه" in doc
    with pytest.raises(ValueError):  # the history bank has no metre questions
        build_exam(bank, "meter:3")
