"""«عروض‌یار»: syllabification, metre table, scanning of famous couplets, rhyme, bot, lexicon."""

import pytest

from myllm.aruz.bot import AruzService
from myllm.aruz.ganjoor import hemistichs, split_rhythm
from myllm.aruz.lexicon import _compatible, learn
from myllm.aruz.meters import METERS, feet_pattern, meter_for_rhythm, same_meter
from myllm.aruz.phonology import Syl, add_izafe, lengthen_final, normalize_verse, split_words, word_readings
from myllm.aruz.rhyme import find_rhyme
from myllm.aruz.scan import Scanner, format_result, meter_index


# ---------------- metres
def test_feet_patterns():
    assert feet_pattern("مفاعیلن مفاعیلن فعولن")[1] == "u---u---u--"
    assert feet_pattern("فعولن فعولن فعولن فعل")[1] == "u--u--u--u-"
    assert feet_pattern("مفعول فاعلات مفاعیل فاعلن")[1] == "--u-u-uu--u-u-"
    assert feet_pattern("مستفعلن مستفعلن")[1] == "--u---u-"
    with pytest.raises(ValueError):
        feet_pattern("فاعلاتون")


def test_ganjoor_labels_map_to_meters():
    m = meter_for_rhythm("فاعلاتن فاعلاتن فاعلن (رمل مسدس محذوف یا وزن مثنوی)")
    assert m.name == "رمل مسدس محذوف"
    rub = meter_for_rhythm("مفعول مفاعیل مفاعیل فعل (وزن رباعی)")
    assert rub.family == "رباعی"
    other_rub = next(x for x in METERS if x.family == "رباعی" and x is not rub)
    assert same_meter(rub, other_rub)
    assert split_rhythm("مفاعیلن ×4 (هزج مثمن سالم)")[1] == "هزج مثمن سالم"


def test_meter_index_contains_ikhtiyarat():
    khafif = next(m for m in METERS if m.name == "خفیف مسدس مخبون")
    ix = meter_index([khafif])
    assert khafif.pattern in ix.variants
    initial_long = "-" + khafif.pattern[1:]  # «فاعلاتن» به جای «فعلاتن»
    assert any(op[0] == "initial_long" for _, _, ops in ix.variants[initial_long] for op in ops)
    taskin_end = khafif.pattern[:-3] + "--"  # «فعلن» -> «فع لن»
    assert taskin_end in ix.variants


# ---------------- phonology
def _best(word):
    return word_readings(word)[0]


@pytest.mark.parametrize("word,weights", [
    ("جان", "-"),      # «ن» after a long vowel does not count
    ("این", "-"),
    ("چون", "-"),
    ("دل", "-"),
    ("خواب", "-u"),    # و معدوله, overlong
    ("ساربان", "-u-"),
    ("نامه", "-u"),     # silent final ه
])
def test_best_readings(word, weights):
    assert _best(word).weights == weights


def test_harakat_force_vowels():
    assert [r.weights for r in word_readings("اَیُّها")] == ["-u-"]  # tashdid doubles ی
    assert _best("خانهٔ").weights == "-uu"  # explicit izafe: khā-ne-ye
    assert "izafe" in _best("زلفِ").tags


def test_izafe_and_lengthening():
    r = _best("دل")
    assert add_izafe(r).weights == "uu"  # del -> de-le
    short = Syl("ک", False, (), "که")
    from myllm.aruz.phonology import Reading

    assert lengthen_final(Reading((short,), 0.0)).weights == "-"
    assert lengthen_final(r) is None  # closed syllable cannot be lengthened


def test_split_words_and_normalize():
    assert split_words("می‌رود  دلم!") == [["می", "رود"], ["دلم"]]
    assert normalize_verse("كتاب علي، «خوب»") == "کتاب علی خوب"


# ---------------- scanning famous couplets (meters verified against Ganjoor labels)
COUPLETS = [
    ("بشنو این نی چون شکایت می‌کند", "از جدایی‌ها حکایت می‌کند", "رمل مسدس محذوف"),
    ("توانا بود هر که دانا بود", "ز دانش دل پیر برنا بود", "متقارب مثمن محذوف"),
    ("ای ساربان آهسته ران کآرام جانم می‌رود", "وان دل که با خود داشتم با دلستانم می‌رود", "رجز مثمن سالم"),
    ("صوفی بیا که آینه صافیست جام را", "تا بنگری صفای می لعل فام را", "مضارع مثمن اخرب مکفوف محذوف"),
    ("دوش دیدم که ملایک در میخانه زدند", "گل آدم بسرشتند و به پیمانه زدند", "رمل مثمن مخبون محذوف"),
    ("ای نام تو بهترین سرآغاز", "بی‌نام تو نامه کی کنم باز", "هزج مسدس اخرب مقبوض محذوف"),
    ("بیا که قصر امل سخت سست بنیاد است", "بیار باده که بنیاد عمر بر باد است", "مجتث مثمن مخبون محذوف"),
    ("بیا تا گل برافشانیم و می در ساغر اندازیم", "فلک را سقف بشکافیم و طرحی نو دراندازیم", "هزج مثمن سالم"),
    ("خداوندا در توفیق بگشای", "نظامی را ره تحقیق بنمای", "هزج مسدس محذوف"),
    ("یوسف گم گشته باز آید به کنعان غم مخور", "کلبه احزان شود روزی گلستان غم مخور", "رمل مثمن محذوف"),
]


@pytest.fixture(scope="module")
def scanner():
    return Scanner()


@pytest.mark.parametrize("first,second,meter", COUPLETS)
def test_famous_couplets(scanner, first, second, meter):
    r1, r2 = scanner.scan_couplet(first, second)
    assert r1.meter is not None and r1.meter.name == meter, format_result(r1)
    assert r2.meter.name == meter
    assert len(r1.syllables) == len(r1.weights)
    assert sum(len(s) for _, s in r1.feet_groups) == len(r1.syllables)


def test_rubai_family(scanner):
    r1, _ = scanner.scan_couplet("این قافله عمر عجب می‌گذرد", "دریاب دمی که با طرب می‌گذرد")
    assert r1.meter.family == "رباعی"


def test_ikhtiyarat_reported(scanner):
    r = scanner.scan("بیا که قصر امل سخت سست بنیاد است")
    assert any("حذف همزه" in n for n in r.ikhtiyarat)
    text = format_result(r)
    assert "وزن:" in text and "تقطیع:" in text and "ارکان:" in text


def test_unscannable_text(scanner):
    r = scanner.scan("سلام")
    assert not r.ok and "وزنی پیدا نشد" in format_result(r)


# ---------------- rhyme
def test_rhyme_and_radif():
    r = find_rhyme(["ای ساربان آهسته ران کآرام جانم می‌رود", "وان دل که با خود داشتم با دلستانم می‌رود"])
    assert r.radif == "میرود" and r.letters == "انم"
    r = find_rhyme(["الا یا ایها الساقی ادر کأسا و ناولها", "که عشق آسان نمود اول ولی افتاد مشکل‌ها",
                    "ز تاب جعد مشکینش چه خون افتاد در دل‌ها"])
    assert r.letters == "لها" and r.radif == ""
    assert find_rhyme(["تابی", "آبی"]).letters == "ابی"  # آ = ا + ā


# ---------------- bot
def test_bot_service():
    s = AruzService(None)
    assert "عروض‌یار" in s.handle_text("/start")
    out = s.handle_text("بشنو این نی چون شکایت می‌کند\nاز جدایی‌ها حکایت می‌کند")
    assert "رمل مسدس محذوف" in out and "قافیه" in out
    poem = s.handle_text("بشنو این نی چون شکایت می‌کند\nاز جدایی‌ها حکایت می‌کند\n"
                         "کز نیستان تا مرا ببریده‌اند\nدر نفیرم مرد و زن نالیده‌اند")
    assert "وزن کل شعر" in poem and "رمل مسدس محذوف" in poem
    assert "حداکثر" in s.handle_text("\n".join(["الف"] * 13))
    assert s.stats["requests"] == 2 and s.stats["poem"] == 1


def test_scan_poem_votes_and_survives_odd_line(scanner):
    lines = ["بشنو این نی چون شکایت می‌کند", "از جدایی‌ها حکایت می‌کند", "سلام"]
    rs = scanner.scan_poem(lines)
    assert rs[0].meter.name == "رمل مسدس محذوف" and len(rs) == 3


# ---------------- ganjoor + lexicon
POEM = {
    "FullUrl": "/test/masnavi/1",
    "Metre": {"Rhythm": "فاعلاتن فاعلاتن فاعلن (رمل مسدس محذوف یا وزن مثنوی)"},
    "Verses": [
        {"Position": "Right", "Text": "بشنو این نی چون شکایت می‌کند", "CoupletIndex": 0},
        {"Position": "Left", "Text": "از جدایی‌ها حکایت می‌کند", "CoupletIndex": 0},
        {"Position": "Right", "Text": "کز نیستان تا مرا ببریده‌اند", "CoupletIndex": 1},
        {"Position": "Left", "Text": "در نفیرم مرد و زن نالیده‌اند", "CoupletIndex": 1},
        {"Position": "Paragraph", "Text": "نثر", "CoupletIndex": 2},
    ],
}


def test_ganjoor_hemistichs():
    hs = hemistichs(POEM)
    assert len(hs) == 4 and hs[0].position == "Right" and hs[0].poet == "test"


def test_lexicon_learning_and_use():
    lex = learn(hemistichs(POEM) * 2, iterations=1, min_count=1, log_every=0)
    assert "کند" in lex and abs(sum(lex["کند"].values()) - 1.0) < 0.05
    sc = Scanner(lex)
    assert sc.scan_couplet(POEM["Verses"][0]["Text"], POEM["Verses"][1]["Text"])[0].meter.name == "رمل مسدس محذوف"
    assert _compatible("V1", "V0") and _compatible("V0", "A0") and not _compatible("A0", "V0")
