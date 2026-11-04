from myllm.data.decontam import Decontaminator
from myllm.data.dedup import dedup_exact, dedup_minhash, jaccard
from myllm.data.formats import (
    from_alpaca,
    from_sharegpt,
    to_messages,
    validate_messages,
)
from myllm.data.langid import guess_language
from myllm.data.licenses import allowed, classify
from myllm.data.normalize_fa import (
    ZWNJ,
    normalize_fa,
    normalize_preserving_code,
    to_ascii_digits,
)
from myllm.data.pii import scrub_pii, valid_iban, valid_luhn, valid_national_id
from myllm.data.quality import HashedQualityClassifier, heuristic_quality


# ---------------- normalization
def test_arabic_letters_become_persian():
    assert normalize_fa("كتاب علي") == "کتاب علی"


def test_digits_modes():
    assert normalize_fa("٣ و 3") == "۳ و 3"  # Arabic-Indic -> Persian, ASCII kept by default
    assert normalize_fa("٣ و 3", digits="ascii") == "3 و 3"
    assert normalize_fa("12", digits="persian") == "۱۲"
    assert to_ascii_digits("۱۴۰۵") == "1405"


def test_zwnj_affixes_and_spacing():
    out = normalize_fa("من   می خواهم کتاب ها را بخوانم .")
    assert out == f"من می{ZWNJ}خواهم کتاب{ZWNJ}ها را بخوانم."


def test_diacritics_tatweel_invisibles():
    assert normalize_fa("سَلامـــ‏") == "سلام"


def test_code_blocks_untouched():
    text = "كد:\n```python\nx = 'ي'  # كامنت\n```"
    out = normalize_preserving_code(text)
    assert out.startswith("کد:") and "x = 'ي'  # كامنت" in out


# ---------------- langid
def test_langid():
    assert guess_language("این یک جمله‌ی فارسی است که برای آزمون نوشته شده.").lang == "fa"
    assert guess_language("This is an English sentence for testing.").lang == "en"
    assert guess_language("هذا النص مكتوب باللغة العربية وهو في الكتاب الذي قرأته.").lang == "ar"
    assert guess_language("def f(x):\n    return x + 1\nprint(f(2))").lang == "code"
    assert guess_language("   ").lang == "unknown"


# ---------------- dedup
def test_exact_dedup_ignores_whitespace_case_zwnj():
    texts = ["Hello  World", "hello world", f"می{ZWNJ}رود", "می رود".replace(" ", ""), "other"]
    assert dedup_exact(texts) == [0, 2, 4]


def test_minhash_near_duplicates():
    base = "مدل‌های زبانی بزرگ با پیش‌بینی توکن بعدی آموزش می‌بینند و کیفیت داده بسیار مهم است. " * 3
    near = base[:-5] + " واقعا"  # small edit at the end (Jaccard ~0.94)
    other = "Completely different text about cooking rice and vegetables for dinner tonight." * 3
    assert dedup_minhash([base, near, other], threshold=0.8) == [0, 2]
    assert jaccard(base, near) > 0.9 > 0.1 > jaccard(base, other)


# ---------------- PII
def test_national_id_checksum():
    assert valid_national_id("0499370899")
    assert not valid_national_id("0499370898")
    assert not valid_national_id("1111111111")


def test_luhn_and_iban():
    assert valid_luhn("4111 1111 1111 1111")
    assert not valid_luhn("4111 1111 1111 1112")
    assert valid_iban("IR062960000000100324200001")
    assert not valid_iban("IR062960000000100324200002")


def test_scrub_pii_persian_digits_and_formats():
    text = "شماره‌ام ۰۹۱۲۳۴۵۶۷۸۹ و ایمیلم a.b@example.com است. کد ملی 0499370899. کارت 4111-1111-1111-1111."
    clean, counts = scrub_pii(text)
    assert "[PHONE]" in clean and "[EMAIL]" in clean and "[NATIONAL_ID]" in clean and "[CARD]" in clean
    assert "۰۹۱۲" not in clean and "example.com" not in clean
    assert counts["PHONE"] == 1 and counts["CARD"] == 1


def test_scrub_pii_keeps_non_pii_numbers():
    text = "سال 1403 بود و 1234567890 تا ستاره دیدیم؛ نسخه 3.11 پایتون."
    clean, counts = scrub_pii(text)
    assert clean == text and not counts  # invalid national-id checksum -> kept


# ---------------- quality
def test_quality_heuristics():
    good = "این یک متن معمولی و مفید فارسی است که درباره‌ی یادگیری ماشین و کاربردهای آن توضیح می‌دهد."
    assert heuristic_quality(good).keep
    spam = "خرید خرید خرید خرید خرید خرید خرید خرید ارزان ارزان ارزان ارزان"
    assert "low_vocabulary" in heuristic_quality(spam).reasons
    assert "too_short" in heuristic_quality("سلام").reasons
    code = "def f(x):\n    return {x: [x, x]}\n" * 3
    assert heuristic_quality(code, is_code=True).keep


def test_hashed_quality_classifier_learns():
    good = ["یادگیری ماشین شاخه‌ای از هوش مصنوعی است.", "فتوسنتز در گیاهان رخ می‌دهد.",
            "شاهنامه اثر فردوسی است.", "آب در صد درجه می‌جوشد."] * 5
    bad = ["کلیک کنید!!! $$$ خرید ارزان $$$", "!!!! $$$ تخفیف $$$ !!!!", "www $$$ ### @@@ !!!", "$$$ $$$ $$$ خرید"] * 5
    clf = HashedQualityClassifier(n_features=2**14).fit(good + bad, [1] * len(good) + [0] * len(bad), epochs=10)
    assert clf.predict_proba("زبان فارسی زبانی زیبا است.") > clf.predict_proba("$$$ خرید ارزان !!! $$$")


# ---------------- decontam
def test_decontamination():
    eval_texts = ["پایتخت استرالیا کجاست و چرا سیدنی پایتخت نیست؟ توضیح بده"]
    deco = Decontaminator(eval_texts, n=6)
    assert deco.is_contaminated("سؤال: پایتخت استرالیا کجاست و چرا سیدنی پایتخت نیست؟ جواب: کانبرا")
    assert not deco.is_contaminated("پایتخت فرانسه پاریس است و برج ایفل در آن قرار دارد.")


# ---------------- formats & licenses
def test_formats():
    a = from_alpaca({"instruction": "جمع کن", "input": "۲ و ۳", "output": "۵"})
    assert a["messages"][0] == {"role": "user", "content": "جمع کن\n\n۲ و ۳"}
    s = from_sharegpt({"conversations": [{"from": "human", "value": "سلام"}, {"from": "gpt", "value": "درود"}]})
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]
    assert to_messages({"prompt": "p", "response": "r", "id": 7})["id"] == 7
    assert validate_messages(s["messages"]) == []
    assert validate_messages([{"role": "assistant", "content": "x"}])


def test_licenses():
    assert classify("Apache-2.0") == "commercial_ok"
    assert classify("cc-by-nc-4.0") == "non_commercial"
    assert classify(None) == "unknown"
    assert not allowed("cc-by-nc-sa-4.0", commercial_safe=True)
    assert allowed("cc-by-nc-sa-4.0", commercial_safe=False)
    assert not allowed("verify", commercial_safe=True, allow_unknown=False)


def test_pipeline_end_to_end():
    from myllm.config import load_config
    from myllm.data.pipeline import run_pipeline
    from myllm.utils import read_jsonl

    rows = read_jsonl("data/samples/raw_dirty.jsonl")
    rows.append({"text": "متنی با لایسنس غیرتجاری که باید حذف شود چون پایپ‌لاین تجاری‌امن است.", "license": "cc-by-nc-4.0"})
    clean, stats = run_pipeline(rows, load_config("configs/smoke/clean.yaml"))
    joined = "\n".join(str(r) for r in clean)
    assert "۰۹۱۲۳۴۵۶۷۸۹" not in joined and "ali.rezaei@example.com" not in joined
    assert stats["dropped"]["license"] == 1
    assert stats["dropped"]["lang_ar"] == 1
    assert stats["dropped"]["exact_dup"] >= 1
    assert "ي" not in joined and "ك" not in joined
    assert any("messages" in r for r in clean)  # conversations survive the pipeline
