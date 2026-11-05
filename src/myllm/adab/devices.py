"""Rule-based literary devices (آرایه‌های ادبی) for a couplet — explainable and precision-first.

    from myllm.adab.devices import find_devices, format_devices
    print(format_devices(find_devices(["دل که آیینه‌ی صافی است غباری دارد", "از خدا می‌طلبم صحبت روشن‌رایی"])))

Detected here (deterministic, each with the evidence words and a school-book explanation):
  جناس تام/ناقص (اختلافی، افزایشی، حرکتی) · تکرار · واج‌آرایی · تضاد · تناسب (مراعات نظیر) · تلمیح ·
  تشبیه با ادات
Not detected here (they need meaning — see the LLM tagger in docs/product/adab.md):
  استعاره، کنایه، مجاز، ایهام، تشخیص، اغراق، حسن تعلیل، لف و نشر

Word positions follow Ganjoor's convention (misra 1/2, words split at whitespace, 1-based), so the
output can be compared with Ganjoor's LiteraryTags (MIT-licensed academic corpus for Hafez).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from myllm.adab.lexicons import (
    ALLUSIONS,
    ANTONYMS,
    FIELDS,
    HOMOGRAPHS,
    SIMILE_BIGRAMS,
    SIMILE_MARKERS,
    STOPWORDS,
    STRICT_FIELDS,
    SUFFIXES,
)
from myllm.aruz.phonology import MARKS, SHORT_MARKS

# baseline letter frequencies (computed once from 200k Ganjoor hemistichs) for واج‌آرایی
LETTER_FREQ = {
    "ر": 0.0898, "ن": 0.0801, "د": 0.0744, "ب": 0.0528, "م": 0.0521, "ت": 0.0448, "س": 0.0331, "ش": 0.0324,
    "ک": 0.0312, "ز": 0.0307, "ل": 0.0218, "گ": 0.021, "خ": 0.0175, "ف": 0.0143, "چ": 0.0116, "ج": 0.0115,
    "ع": 0.0091, "پ": 0.0087, "ق": 0.0084, "ح": 0.0065, "ص": 0.0043, "غ": 0.0041, "ط": 0.0035,
}
_PUNCT = "،؛؟!«»:\"'.,;?()[]-–—…"
_FIX = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "‍": ""})


@dataclass(frozen=True)
class Word:
    misra: int  # 1 or 2
    index: int  # 1-based position in the hemistich (whitespace split)
    raw: str  # as written (with harakat)
    plain: str  # letters only, no ZWNJ

    @property
    def content(self) -> bool:
        """A lexical word: not a function word, not a function word + enclitic («آنم», «ازش»)."""
        return len(self.plain) >= 2 and not _function_word(self.plain)


# function words that take pronoun enclitics («آنم», «اینت», «درش»); «دوست», «گوش» stay content words
_ENCLITIC_HOSTS = {"آن", "این", "از", "به", "در", "با", "بر", "که", "چه", "تو", "من", "او", "ما", "خود", "همه",
                   "هم", "نه", "چون", "چو", "ز", "کز", "وز", "زان", "زین", "کان", "کاین", "وان", "وین"}
_ENCLITICS = ("م", "ت", "ش", "مان", "تان", "شان", "ست", "یم", "ید", "ند")


def _function_word(plain: str) -> bool:
    if plain in STOPWORDS:
        return True
    return any(plain.endswith(e) and plain[: -len(e)] in _ENCLITIC_HOSTS for e in _ENCLITICS)


@dataclass
class Device:
    kind: str  # e.g. "جناس ناقص اختلافی"
    words: list[Word]
    explanation: str
    confidence: str = "بالا"  # بالا | متوسط
    extra: dict = field(default_factory=dict)

    def positions(self) -> set[tuple[int, int]]:
        return {(w.misra, w.index) for w in self.words}


def tokenize(misras: list[str]) -> list[Word]:
    out = []
    for m, text in enumerate(misras, 1):
        for i, tok in enumerate(text.translate(_FIX).split(), 1):
            raw = tok.strip(_PUNCT)
            plain = "".join(ch for ch in raw if ch not in MARKS and ch != "‌")
            if plain:
                out.append(Word(m, i, raw, plain))
    return out


def stems(word: str) -> set[str]:
    forms = {word}
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            forms.add(word[: -len(suf)])
    if word.startswith("می") and len(word) > 4:
        forms.add(word[2:])
    return forms


def _vowel_map(raw: str) -> dict[int, str]:
    """letter index -> the short-vowel mark written on it (ZWNJ and other marks ignored)."""
    out, k = {}, -1
    for ch in raw:
        if ch == "‌":
            continue
        if ch in MARKS:
            if ch in SHORT_MARKS and k >= 0:
                out[k] = ch
            continue
        k += 1
    return out


# --- جناس ---------------------------------------------------------------------------------------------
# final ه is usually a new word (باد/باده، چشم/چشمه، خار/خاره) except on a past stem (رفت/رفته)
_GRAMMATICAL_END = {"ی", "م", "ت", "ش", "ن", "ا", "د"}
_GRAMMATICAL_START = {"ب", "ن"}  # not م: محرم/حرم، منظر/نظر، مقدم/قدم are classic جناس
_PAST_STEMS = {"رفت", "گفت", "کرد", "شد", "بود", "آمد", "داد", "خورد", "برد", "زد", "کشید", "رسید", "گذشت", "نشست",
               "گرفت", "یافت", "شنید", "ماند", "افتاد", "نهاد", "خواست", "توانست", "دانست", "سوخت", "ساخت", "دید",
               "شکست", "بست", "پیوست", "گشت", "کشت", "داشت", "گذاشت", "نوشت", "شست", "خرید", "کوشید", "پوشید",
               "بخشید", "آفرید", "دوید", "پرید", "خندید", "گریست", "نگریست", "آمیخت", "ریخت", "آموخت", "دوخت",
               "فروخت", "انداخت", "شناخت", "تاخت", "باخت", "پرداخت", "نواخت", "خفت", "شکفت", "پذیرفت", "نهفت",
               "آشفت", "خاست", "آراست", "جست", "رست", "پرسید", "رمید", "چکید", "تابید", "گردید", "نمود", "فزود"}
# present stems / imperatives: «مبین/بین»، «بخور/مخور»، «برفت/نرفت» are one verb with a prefix, not جناس
_PRESENT_STEMS = {"بین", "رو", "ده", "زن", "خور", "کش", "نه", "گیر", "پوش", "ساز", "ریز", "بخش", "نوش", "پرس",
                  "خوان", "ران", "مان", "دان", "شنو", "گو", "کن", "شو", "باش", "آ", "آور", "آر", "بر", "جو", "بند",
                  "سوز", "رس", "گذار", "گذر", "فشان", "نشین", "خیز", "پوی", "گری", "خند", "بار", "کار", "دار",
                  "گشا", "نما", "نویس", "فرست", "افکن", "فکن", "یاب", "آموز", "تاب", "پیچ", "شکن", "خسب", "خواه",
                  "گرد", "گوی", "روی", "جوی", "شوی", "آی", "نگر", "بخشای"}
_VERB_STEMS = _PRESENT_STEMS | _PAST_STEMS
_VARIANT_PAIRS = {frozenset(p) for p in (("نیک", "نیکو"), ("گیس", "گیسو"), ("سپید", "سفید"), ("خموش", "خاموش"))}
_PERSONAL_ENDINGS = {"ی", "م", "یم", "ید", "ند", "ه", "ن"}
_VERB_SUFFIXES = ("ییم", "یید", "یند", "یم", "ید", "ند", "ی", "م", "د", "ه", "ن", "")


def _prefixed_verb(word: str) -> bool:
    """ب/ن/م + a verb stem + a personal ending («برفت»، «مخور»، «برود»، «نگوییم»)."""
    if len(word) < 3 or word[0] not in "بنم":
        return False
    rest = word[1:]
    return any(rest.endswith(e) and rest[: len(rest) - len(e)] in _VERB_STEMS for e in _VERB_SUFFIXES)
_VOWEL_LETTERS = {"ا", "آ", "و", "ی", "ه"}
# light / auxiliary verbs (کردن، شدن، بودن): «نبود/نشود»، «کم/کن» are not a poet's جناس
_LIGHT_VERBS = {"کن", "کرد", "شو", "شد", "بود", "باش", "هست", "است", "نیست", "کرده", "شده", "بوده"}
_VERB_PREFIXES = ("نمی", "همی", "می", "ن", "ب")
_VERB_ENDINGS = ("یم", "ید", "ند", "ی", "م", "د", "ه", "ن")


def _light_verb(word: str) -> bool:
    cands = {word}
    for p in _VERB_PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 2:
            cands.add(word[len(p) :])
    for c in list(cands):
        for e in _VERB_ENDINGS:
            if c.endswith(e) and len(c) - len(e) >= 2:
                cands.add(c[: -len(e)])
    return bool(cands & _LIGHT_VERBS)


def _spelling_variant(x: str, y: str) -> bool:
    """Same word, two spellings — never جناس:
    راه/ره، سیاه/سیه (ا before ه) · آورد/آرد، بیاور/بیار، اوفتاد/افتاد (dropped و) · آیینه/آینه (ی/یی) ·
    افتاد/فتاد، افکند/فکند (dropped initial ا) · ملامت/ملامه (ت/ه)."""
    if frozenset((x, y)) in _VARIANT_PAIRS:
        return True
    short, long_ = (x, y) if len(x) < len(y) else (y, x)
    if len(long_) == len(short) + 1:
        for k in range(len(long_)):
            if long_[:k] + long_[k + 1 :] != short:
                continue
            c, head = long_[k], long_[:k]
            nxt = long_[k + 1] if k + 1 < len(long_) else ""
            if c == "ا" and nxt == "ه":
                return True
            if c == "و" and nxt in ("ر", "ف") and (head.endswith("آ") or head in ("ا", "بیا", "نیا", "میا")):
                return True
            if c == "ی" and "ی" in (head[-1:], nxt):
                return True
            if k == 0 and c == "ا" and nxt == "ف":
                return True
    if len(x) == len(y) and x[:-1] == y[:-1] and {x[-1], y[-1]} == {"ت", "ه"}:
        return True
    return False


def _jenas_type(a: Word, b: Word) -> tuple[str, str] | None:
    x, y = a.plain, b.plain
    if _spelling_variant(x, y):
        return None
    if x == y:
        # only vowels written on both words count; the last letter's mark is usually the اضافه kasra
        va, vb = _vowel_map(a.raw), _vowel_map(b.raw)
        if any(va[k] != vb[k] for k in va.keys() & vb.keys() if k != len(x) - 1):
            return "جناس ناقص حرکتی", f"«{a.raw}» و «{b.raw}» حروف یکسان و حرکت متفاوت دارند."
        if x in HOMOGRAPHS:
            return ("جناس تام (اگر معنی دو واژه فرق کند)",
                    f"«{x}» دو بار آمده؛ اگر در دو معنای متفاوت باشد جناس تام است، وگرنه تکرار.")
        return None
    if len(x) == len(y):
        diff = [k for k in range(len(x)) if x[k] != y[k]]
        if len(diff) != 1 or len(x) < 2:
            return None
        k = diff[0]
        pair = {x[k], y[k]}
        if k == 0 and "آ" in pair:
            return None  # آب/شب: initial آ is a long vowel, so the words differ in more than one sound
        if k == 0 and (_prefixed_verb(x) or _prefixed_verb(y)):
            return None  # بخور/مخور، برفت/گرفت: the "different letter" is a verbal prefix
        # (initial ا is a glottal stop + short vowel, so امان/زمان and امل/عمل do differ in one sound)
        if k > 0 and pair & {"ا", "آ"} and not pair <= _VOWEL_LETTERS:
            return None  # دلم/دام: a long vowel against a consonant changes the syllables, not one sound
        where = "آغاز" if k == 0 else ("پایان" if k == len(x) - 1 else "میان")
        return ("جناس ناقص اختلافی",
                f"«{x}» و «{y}» فقط در یک حرف ({where} واژه: «{x[k]}»/«{y[k]}») فرق دارند.")
    short, long_ = (x, y) if len(x) < len(y) else (y, x)
    if len(long_) - len(short) != 1 or len(short) < 2:
        return None
    if long_.startswith(short) and (long_[-1] in _GRAMMATICAL_END or (long_[-1] == "ه" and short in _PAST_STEMS)):
        return None  # دل/دلم، یار/یارا، بوی/بویی، رفت/رفته: a grammatical ending, not جناس
    if long_.endswith(short) and (long_[0] in _GRAMMATICAL_START or _prefixed_verb(long_)):
        return None  # رفت/برفت، دید/ندید، بین/مبین: verbal prefix
    for k in range(len(long_)):
        if long_[:k] + long_[k + 1 :] == short:
            extra = long_[k]
            if extra in ("د", "ت") and k >= 2 and long_[k + 1 :] in _PERSONAL_ENDINGS:
                return None  # فکنی/فکندی: present and past stem of one verb
            where = "آغاز" if k == 0 else ("پایان" if k == len(long_) - 1 else "میان")
            return ("جناس ناقص افزایشی", f"«{long_}» یک حرف ({where}: «{extra}») بیشتر از «{short}» دارد.")
    return None


_INFLECTION = set("ماییتشندو")


def _inflectional_pair(x: str, y: str) -> bool:
    """«گفتم/گفتا», «کرده/کردی»: same stem, only inflectional endings differ -> not جناس."""
    if len(x) != len(y) or len(x) < 3:
        return False
    return x[:-1] == y[:-1] and x[-1] in _INFLECTION and y[-1] in _INFLECTION


def _salience(a: Word, b: Word) -> float:
    """Higher = more likely the poet's deliberate جناس: close together, longer words, parallel positions."""
    dist = abs(a.index - b.index) if a.misra == b.misra else 99
    score = 1.0 if dist <= 2 else (0.6 if dist <= 4 else 0.3)
    if a.misra != b.misra and abs(a.index - b.index) <= 1:
        score += 0.5  # same position in both hemistichs (parallelism)
    score += 0.15 * (min(len(a.plain), len(b.plain)) - 2)
    return round(score, 2)


def find_jenas(words: list[Word], exclude: set[str] | None = None) -> list[Device]:
    out, seen = [], set()
    exclude = exclude or set()
    cw = [w for w in words if w.content and w.plain not in exclude and not _light_verb(w.plain)]
    for i in range(len(cw)):
        for j in range(i + 1, len(cw)):
            a, b = cw[i], cw[j]
            key = tuple(sorted((a.plain, b.plain)))
            if key in seen:
                continue
            if _inflectional_pair(a.plain, b.plain):
                continue
            t = _jenas_type(a, b)
            if t:
                seen.add(key)
                out.append(Device(t[0], [a, b], t[1], "متوسط" if "اگر" in t[0] else "بالا",
                                  {"salience": _salience(a, b)}))
    out.sort(key=lambda d: -d.extra.get("salience", 0))
    return out


# --- تکرار ---------------------------------------------------------------------------------------------
def find_repetition(words: list[Word], radif: set[str]) -> list[Device]:
    groups: dict[str, list[Word]] = {}
    for w in words:
        if w.content and w.plain not in radif:
            groups.setdefault(w.plain, []).append(w)
    return [Device("تکرار", ws, f"واژه‌ی «{p}» {len(ws)} بار تکرار شده است.") for p, ws in groups.items() if len(ws) >= 2]


# --- واج‌آرایی --------------------------------------------------------------------------------------------
def find_alliteration(words: list[Word], min_count: int = 5, ratio: float = 2.2) -> list[Device]:
    letters = "".join(w.plain for w in words)
    total = len(letters)
    if total < 20:
        return []
    c = Counter(ch for ch in letters if ch in LETTER_FREQ)
    out = []
    for ch, n in c.most_common(3):
        expected = LETTER_FREQ[ch] * total
        if n >= min_count and n >= ratio * expected:
            hits = [w for w in words if ch in w.plain]
            out.append(Device("واج‌آرایی", hits, f"صامت «{ch}» {n} بار آمده (حدود {n / expected:.1f} برابر حالت عادی) "
                              "و موسیقی درونی ساخته است.", extra={"letter": ch, "count": n}))
    return out


# --- تضاد ---------------------------------------------------------------------------------------------
_ANT = {}
for _a, _b in ANTONYMS:
    _ANT.setdefault(_a, set()).add(_b)
    _ANT.setdefault(_b, set()).add(_a)


def find_antithesis(words: list[Word]) -> list[Device]:
    out, seen = [], set()
    forms = [(w, stems(w.plain)) for w in words if len(w.plain) >= 1]
    for i, (a, fa) in enumerate(forms):
        for b, fb in forms[i + 1 :]:
            for x in fa:
                hit = next((y for y in fb if y in _ANT.get(x, ())), None)
                if hit:
                    key = tuple(sorted((x, hit)))
                    if key not in seen:
                        seen.add(key)
                        out.append(Device("تضاد", [a, b], f"«{x}» و «{hit}» معنای متضاد دارند."))
                    break
    return out


# --- تناسب (مراعات نظیر) ------------------------------------------------------------------------------
def find_harmony(words: list[Word]) -> list[Device]:
    out = []
    for name, members in FIELDS.items():
        hits, found = [], set()
        for w in words:
            m = next((s for s in stems(w.plain) if s in members), None)
            if m and m not in found:
                found.add(m)
                hits.append(w)
        need = 3 if name in STRICT_FIELDS else 2
        if len(found) >= need:
            out.append(Device("تناسب (مراعات نظیر)", hits,
                              f"واژه‌های {'، '.join('«' + f + '»' for f in sorted(found))} از یک مجموعه‌اند ({name}).",
                              "بالا" if len(found) >= 3 else "متوسط", {"field": name}))
    return out


# --- تلمیح --------------------------------------------------------------------------------------------
def find_allusion(words: list[Word]) -> list[Device]:
    present = set()
    for w in words:
        present |= stems(w.plain)
    joined = " ".join(w.plain for w in words)
    out, seen = [], set()
    for w in words:
        for s in stems(w.plain):
            if s in ALLUSIONS and s not in seen:
                note, companions = ALLUSIONS[s]
                if companions and not (companions & present or any(c in joined for c in companions)):
                    continue
                seen.add(s)
                out.append(Device("تلمیح", [w], f"اشاره به «{s}»: {note}"))
    if "آب" in present and ({"حیات", "حیوان", "زندگانی"} & present):
        out.append(Device("تلمیح", [w for w in words if w.plain in ("آب", "حیات", "حیوان", "زندگانی")],
                          "اشاره به «آب حیات»: چشمه‌ی جاودانگی که خضر از آن نوشید و اسکندر نیافت."))
    return out


# --- تشبیه با ادات --------------------------------------------------------------------------------------
def find_simile(words: list[Word]) -> list[Device]:
    out = []
    for k, w in enumerate(words):
        marker = w.plain if w.plain in SIMILE_MARKERS else None
        nxt = k + 1
        if marker is None and k + 1 < len(words) and (w.plain, words[k + 1].plain) in SIMILE_BIGRAMS:
            marker, nxt = f"{w.plain} {words[k + 1].plain}", k + 2
        if marker is None:
            continue
        last_in_misra = k + 1 >= len(words) or words[k + 1].misra != w.misra
        prev = words[k - 1].plain if k and words[k - 1].misra == w.misra else ""
        if marker == "مانند" and (last_in_misra or prev in ("در", "وا")):
            continue  # «درمانند/در مانند» = they remain (verb)
        if marker == "مثل" and prev == "به":
            continue  # «به مَثَل» = for example
        tail = words[nxt] if nxt < len(words) and words[nxt].misra == w.misra else None
        target = f"؛ مشبه‌به احتمالاً «{tail.plain}» است" if tail is not None and tail.content else ""
        out.append(Device("تشبیه", [w] + ([tail] if tail is not None else []),
                          f"ادات تشبیه «{marker}» آمده{target}.", "بالا",
                          {"marker": marker}))
    return out


def find_devices(misras: list[str]) -> list[Device]:
    from myllm.aruz.rhyme import find_rhyme

    words = tokenize(misras)
    radif = set(find_rhyme(misras).radif.split()) if len(misras) == 2 else set()
    jenas = find_jenas(words, exclude=radif)
    tam = {d.words[0].plain for d in jenas if d.kind.startswith("جناس تام")}
    repetition = [d for d in find_repetition(words, radif) if d.words[0].plain not in tam]
    return (jenas + repetition + find_antithesis(words) + find_harmony(words) + find_allusion(words)
            + find_simile(words) + find_alliteration(words))


def format_devices(devices: list[Device]) -> str:
    if not devices:
        return ("آرایه‌ی لفظیِ قطعی پیدا نکردم. (آرایه‌های معنایی مثل استعاره، کنایه، مجاز و ایهام را این بخش تشخیص نمی‌دهد.)")
    lines = []
    for d in devices:
        conf = "" if d.confidence == "بالا" else " (احتمالی)"
        lines.append(f"• {d.kind}{conf}: {d.explanation}")
    lines.append("ℹ️ استعاره، کنایه، مجاز و ایهام نیاز به درک معنا دارند و در این بخش بررسی نمی‌شوند.")
    return "\n".join(lines)
