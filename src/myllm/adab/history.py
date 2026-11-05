"""Literary-history question bank (تاریخ ادبیات) — curated canon + Ganjoor verses.

    python -m myllm.adab.history build --n 400                    # -> data/processed/history_bank.jsonl
    python -m myllm.adab.history check                            # canon death dates vs Ganjoor's poet.json

Why a hand-curated canon instead of Ganjoor's category titles: Ganjoor lists every work *attributed*
to a poet (e.g. «هیلاج‌نامه», «جوهرالذات» under عطار), which scholars and the school books reject.
Exam questions must be right, so works, centuries, styles and titles come from the list below
(textbook-level facts; disputed facts are left out — e.g. نظامی's death year spans two centuries,
so no century question is asked about him). Ganjoor is used for the verses («این بیت از کیست؟»)
and as an independent cross-check of the death dates (``check``).

Question kinds: work_author, work_of, not_by, century, style, chronology, title, verse_poet.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from myllm.aruz.exercises import Question

STYLES = ("سبک خراسانی", "سبک عراقی", "سبک هندی", "دوره‌ی بازگشت ادبی", "دوره‌ی بیداری (مشروطه)", "دوره‌ی معاصر")
KH, ER, HE, BA, MA, MO = STYLES
ROOTS = ("data/raw/ganjoor", "data/raw/ganjoor_newpoets", "data/raw/ganjoor_test2")


@dataclass(frozen=True)
class Author:
    name: str
    slug: str | None = None  # Ganjoor slug (verses, date cross-check)
    death: int | None = None  # lunar Hijri; None when sources disagree across centuries or for modern authors
    style: str | None = None  # None when transitional or classified differently by different books
    works: tuple[str, ...] = ()
    title: str | None = None  # a well-known لقب
    modern: bool = False
    verse: bool = True  # False for prose authors (no «این بیت از کیست؟»)
    pen: tuple[str, ...] = ()  # تخلص / names that give the answer away inside a verse


A = Author
CANON: tuple[Author, ...] = (
    # --- classical poets -----------------------------------------------------------------------------
    A("رودکی", "roodaki", 329, KH, title="پدر شعر فارسی", pen=("رودکی",)),
    A("فردوسی", "ferdousi", 411, KH, ("شاهنامه",), "حکیم توس", pen=("فردوسی",)),
    A("فرخی سیستانی", "farrokhi", 429, KH, pen=("فرخی",)),
    A("عنصری", "onsori", 431, KH, pen=("عنصری",)),
    A("منوچهری دامغانی", "manoochehri", 432, KH, pen=("منوچهری",)),
    A("اسدی توسی", "asadi", 465, KH, ("گرشاسب‌نامه", "لغت فرس"), pen=("اسدی",)),
    A("ناصرخسرو", "naserkhosro", 481, KH, ("سفرنامه", "زادالمسافرین", "وجه دین", "خوان‌الاخوان", "جامع‌الحکمتین",
                                           "روشنایی‌نامه"), "حجت خراسان", pen=("حجت", "ناصر")),
    A("فخرالدین اسعد گرگانی", "asad", None, KH, ("ویس و رامین",)),
    A("مسعود سعد سلمان", "masood", 515, KH, pen=("مسعود",)),
    A("سنایی", "sanaee", 525, None, ("حدیقة‌الحقیقه", "سیرالعباد الی المعاد", "کارنامه‌ی بلخ"), pen=("سنایی",)),
    A("انوری", "anvari", 585, KH, pen=("انوری",)),
    A("خاقانی", "khaghani", 595, None, ("تحفة‌العراقین",), "حسان‌العجم", pen=("خاقانی",)),
    A("نظامی گنجوی", "nezami", None, None, ("مخزن‌الاسرار", "خسرو و شیرین", "لیلی و مجنون", "هفت پیکر", "اسکندرنامه"),
      "حکیم گنجه", pen=("نظامی",)),
    A("عطار نیشابوری", "attar", 618, None, ("منطق‌الطیر", "الهی‌نامه", "اسرارنامه", "مصیبت‌نامه", "مختارنامه",
                                           "تذکرة‌الاولیاء"), pen=("عطار",)),
    A("کمال‌الدین اسماعیل", "kamal", 635, None, title="خلاق‌المعانی", pen=("کمال",)),
    A("مولوی", "moulavi", 672, ER, ("مثنوی معنوی", "دیوان شمس", "فیه ما فیه", "مکتوبات", "مجالس سبعه"),
      pen=("مولانا", "شمس", "خموش", "خمش")),
    A("فخرالدین عراقی", "eraghi", 688, ER, ("لمعات", "عشاق‌نامه"), pen=("عراقی",)),
    A("سعدی", "saadi", 691, ER, ("بوستان", "گلستان"), "افصح‌المتکلمین", pen=("سعدی",)),
    A("سلطان ولد", "valad", 712, None, ("ولدنامه", "رباب‌نامه"), pen=("ولد",)),
    A("شیخ محمود شبستری", "shabestari", 720, ER, ("گلشن راز",)),
    A("امیرخسرو دهلوی", "khosro", 725, None, ("مطلع‌الانوار", "شیرین و خسرو", "مجنون و لیلی", "آیینه‌ی اسکندری",
                                             "هشت بهشت"), pen=("خسرو",)),
    A("اوحدی مراغه‌ای", "ouhadi", 738, ER, ("جام جم", "منطق‌العشاق"), pen=("اوحدی",)),
    A("خواجوی کرمانی", "khajoo", 750, ER, ("همای و همایون", "گل و نوروز", "روضة‌الانوار", "کمال‌نامه", "گوهرنامه"),
      "نخلبند شعرا", pen=("خواجو",)),
    A("ابن یمین", "ebneyamin", 769, ER, pen=("یمین",)),
    A("عبید زاکانی", "obeyd", 772, ER, ("اخلاق‌الاشراف", "رساله‌ی دلگشا", "صد پند"), pen=("عبید",)),
    A("سلمان ساوجی", "salman", 778, ER, ("جمشید و خورشید", "فراق‌نامه"), pen=("سلمان",)),
    A("حافظ", "hafez", 792, ER, title="لسان‌الغیب", pen=("حافظ",)),
    A("شاه نعمت‌الله ولی", "shahnematollah", 834, None, pen=("نعمت‌الله", "سید")),
    A("ابن حسام خوسفی", "ebnehesam", 875, None, ("خاوران‌نامه",), pen=("ابن حسام",)),
    A("جامی", "jami", 898, ER, ("هفت اورنگ", "بهارستان", "نفحات‌الانس", "لوایح", "سلامان و ابسال", "یوسف و زلیخا",
                                "تحفة‌الاحرار", "سبحة‌الابرار", "سلسلة‌الذهب", "لیلی و مجنون", "خردنامه‌ی اسکندری"),
      "خاتم‌الشعرا", pen=("جامی",)),
    A("وحشی بافقی", "vahshi", 991, None, ("فرهاد و شیرین", "ناظر و منظور", "خلد برین"), pen=("وحشی",)),
    A("محتشم کاشانی", "mohtasham", 996, None, pen=("محتشم",)),
    A("عرفی شیرازی", "orfi", 999, HE, pen=("عرفی",)),
    A("شیخ بهایی", "bahaee", 1030, None, ("نان و حلوا", "شیر و شکر", "کشکول", "جامع عباسی"), pen=("بهایی",)),
    A("کلیم کاشانی", "kalim", 1061, HE, pen=("کلیم",)),
    A("صائب تبریزی", "saeb", 1086, HE, pen=("صائب",)),
    A("بیدل دهلوی", "bidel", 1133, HE, ("چهار عنصر", "طور معرفت", "محیط اعظم"), "ابوالمعانی", pen=("بیدل",)),
    A("هاتف اصفهانی", "hatef", 1198, BA, pen=("هاتف",)),
    A("ملا احمد نراقی", "naraghi", 1245, None, ("طاقدیس",), pen=("صفایی",)),
    A("قائم‌مقام فراهانی", "ghaemmagham", 1251, None, ("منشآت", "جلایرنامه"), pen=("ثنایی",)),
    A("قاآنی", "ghaani", 1270, BA, ("پریشان",), pen=("قاآنی",)),
    A("فروغی بسطامی", "forooghi", 1274, BA, pen=("فروغی",)),
    A("رضاقلی‌خان هدایت", "rhedayat", 1288, None, ("مجمع‌الفصحا", "ریاض‌العارفین"), pen=("هدایت",)),
    A("عمان سامانی", "omman", 1322, None, ("گنجینة‌الاسرار",), pen=("عمان",)),
    # --- classical prose -----------------------------------------------------------------------------
    A("ابوالفضل بیهقی", "beyhaghi", 470, None, ("تاریخ بیهقی",), verse=False),
    A("هجویری", "hojviri", None, None, ("کشف‌المحجوب",), verse=False),
    A("خواجه عبدالله انصاری", "abdullah", 481, None, ("مناجات‌نامه", "صد میدان", "طبقات‌الصوفیه"), verse=False),
    A("عنصرالمعالی کیکاووس", "keikavus", None, None, ("قابوس‌نامه",), verse=False),
    A("خواجه نظام‌الملک", None, 485, None, ("سیاست‌نامه",), verse=False),
    A("امام محمد غزالی", "ghazzali", 505, None, ("کیمیای سعادت", "نصیحة‌الملوک"), verse=False),
    A("عین‌القضات همدانی", "eynolghozat", 525, None, ("تمهیدات",), verse=False),
    A("رشیدالدین میبدی", "meybodi", None, None, ("کشف‌الاسرار و عدة‌الابرار",), verse=False),
    A("نظامی عروضی", "aroozi", None, None, ("چهار مقاله",), verse=False),
    A("نصرالله منشی", "monshi", None, None, ("کلیله و دمنه",), verse=False),
    A("حمیدالدین بلخی", "hamid", 559, None, ("مقامات حمیدی",), verse=False),
    A("محمد بن منور", "mmon", None, None, ("اسرارالتوحید",), verse=False),
    A("بهاء ولد", "baha", 628, None, ("معارف",), verse=False),
    A("نجم‌الدین رازی", "daye", 654, None, ("مرصادالعباد",), verse=False),
    A("سعدالدین وراوینی", "varavini", None, None, ("مرزبان‌نامه",), verse=False),
    A("خواجه نصیرالدین طوسی", "khajenasir", 672, None, ("اخلاق ناصری", "اساس‌الاقتباس"), verse=False),
    A("عطاملک جوینی", None, 681, None, ("تاریخ جهانگشا",), verse=False),
    A("ملاحسین واعظ کاشفی", None, 910, None, ("انوار سهیلی", "روضة‌الشهدا", "اخلاق محسنی"), verse=False),
    # --- awakening (مشروطه) and contemporary ---------------------------------------------------------
    A("ادیب‌الممالک فراهانی", "adib", style=MA, modern=True, pen=("امیری", "ادیب")),
    A("ایرج میرزا", "iraj", style=MA, works=("زهره و منوچهر", "عارف‌نامه"), modern=True, pen=("ایرج",)),
    A("میرزاده عشقی", "eshghi", style=MA, works=("رستاخیز سلاطین ایران",), modern=True, pen=("عشقی",)),
    A("عارف قزوینی", "aref", style=MA, modern=True, pen=("عارف",)),
    A("فرخی یزدی", "yazdi", style=MA, modern=True, pen=("فرخی",)),
    A("ملک‌الشعرا بهار", "bahar", style=MA, works=("سبک‌شناسی",), modern=True, pen=("بهار",)),
    A("علی‌اکبر دهخدا", None, style=MA, works=("چرند و پرند", "امثال و حکم", "لغت‌نامه"), modern=True, verse=False),
    A("پروین اعتصامی", "parvin", modern=True, pen=("پروین",)),
    A("اقبال لاهوری", "iqbal", works=("اسرار خودی", "رموز بی‌خودی", "جاویدنامه", "پیام مشرق", "زبور عجم"), modern=True,
      pen=("اقبال",)),
    A("رهی معیری", "rahi", style=MO, works=("سایه‌ی عمر",), modern=True, pen=("رهی",)),
    A("شهریار", "shahriar", style=MO, works=("حیدربابایه سلام",), modern=True, pen=("شهریار",)),
    A("نیما یوشیج", style=MO, works=("افسانه", "ققنوس", "مانلی"), title="پدر شعر نو", modern=True),
    A("احمد شاملو", style=MO, works=("هوای تازه", "آیدا در آینه", "ققنوس در باران", "باغ آینه", "کتاب کوچه"),
      modern=True),
    A("مهدی اخوان ثالث", style=MO, works=("زمستان", "آخر شاهنامه", "از این اوستا", "ارغنون"), modern=True),
    A("سهراب سپهری", style=MO, works=("هشت کتاب", "صدای پای آب", "مسافر", "حجم سبز", "اتاق آبی"), modern=True),
    A("فروغ فرخزاد", style=MO, works=("تولدی دیگر", "ایمان بیاوریم به آغاز فصل سرد", "اسیر", "دیوار", "عصیان"),
      modern=True),
    A("هوشنگ ابتهاج", style=MO, works=("سیاه‌مشق", "سراب", "یادگار خون سرو"), modern=True),
    A("محمدرضا شفیعی کدکنی", style=MO, works=("آینه‌ای برای صداها", "هزاره‌ی دوم آهوی کوهی", "موسیقی شعر",
                                              "صور خیال در شعر فارسی", "زمزمه‌ها"), modern=True),
    A("قیصر امین‌پور", style=MO, works=("تنفس صبح", "آینه‌های ناگهان", "گل‌ها همه آفتابگردانند", "دستور زبان عشق",
                                        "در کوچه آفتاب", "ظهر روز دهم"), modern=True),
    A("سلمان هراتی", style=MO, works=("از آسمان سبز", "دری به خانه‌ی خورشید", "از این ستاره تا آن ستاره"), modern=True),
    A("سید حسن حسینی", style=MO, works=("هم‌صدا با حلق اسماعیل", "گنجشک و جبرئیل"), modern=True),
    A("طاهره صفارزاده", style=MO, works=("طنین در دلتا", "سد و بازوان", "مردان منحنی"), modern=True),
    A("علی موسوی گرمارودی", style=MO, works=("خط خون", "سرود رگبار", "عبور"), modern=True),
    A("منوچهر آتشی", style=MO, works=("آهنگ دیگر", "وصف گل سوری"), modern=True),
    A("محمدعلی جمالزاده", works=("یکی بود یکی نبود",), title="پدر داستان کوتاه فارسی", modern=True, verse=False),
    A("صادق هدایت", style=MO, works=("بوف کور", "سه قطره خون", "سگ ولگرد", "زنده به گور", "حاجی آقا", "علویه خانم"),
      modern=True, verse=False),
    A("بزرگ علوی", style=MO, works=("چشم‌هایش",), modern=True, verse=False),
    A("جلال آل‌احمد", style=MO, works=("مدیر مدرسه", "غرب‌زدگی", "نون و القلم", "سنگی بر گوری", "خسی در میقات",
                                       "سه تار", "زن زیادی", "نفرین زمین"), modern=True, verse=False),
    A("سیمین دانشور", style=MO, works=("سووشون", "جزیره‌ی سرگردانی", "به کی سلام کنم"), modern=True, verse=False),
    A("محمود دولت‌آبادی", style=MO, works=("کلیدر", "جای خالی سلوچ", "روزگار سپری‌شده‌ی مردم سالخورده"), modern=True,
      verse=False),
    A("صمد بهرنگی", style=MO, works=("ماهی سیاه کوچولو", "اولدوز و کلاغ‌ها"), modern=True, verse=False),
    A("غلامحسین ساعدی", style=MO, works=("عزاداران بیل", "ترس و لرز"), modern=True, verse=False),
    A("علی شریعتی", style=MO, works=("کویر",), modern=True, verse=False),
    A("عبدالحسین زرین‌کوب", style=MO, works=("از کوچه‌ی رندان", "پله پله تا ملاقات خدا", "سر نی", "با کاروان حله",
                                             "نقد ادبی"), modern=True, verse=False),
    A("محمدعلی اسلامی ندوشن", style=MO, works=("روزها", "ایران را از یاد نبریم"), modern=True, verse=False),
    A("نادر ابراهیمی", style=MO, works=("بار دیگر شهری که دوست می‌داشتم", "آتش بدون دود", "یک عاشقانه‌ی آرام"),
      modern=True, verse=False),
    A("رسول پرویزی", style=MO, works=("شلوارهای وصله‌دار",), modern=True, verse=False),
    A("محمد بهمن‌بیگی", style=MO, works=("بخارای من ایل من",), modern=True, verse=False),
)

ORDINALS = {3: "سوم", 4: "چهارم", 5: "پنجم", 6: "ششم", 7: "هفتم", 8: "هشتم", 9: "نهم", 10: "دهم", 11: "یازدهم",
            12: "دوازدهم", 13: "سیزدهم", 14: "چهاردهم"}


def century(year: int) -> int:
    return (year - 1) // 100 + 1


def century_label(c: int) -> str:
    return f"قرن {ORDINALS[c]}"


def authors_of(canon: tuple[Author, ...] = CANON) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for a in canon:
        for w in a.works:
            out.setdefault(w, set()).add(a.name)
    return out


def _mcq(kind: str, prompt: str, correct: str, wrong: list[str], rng: random.Random, explanation: str,
         verse: list[str] | None = None, source: str = "") -> Question:
    options = [correct] + wrong
    rng.shuffle(options)
    return Question(kind, prompt, verse or [], options, options.index(correct), correct, explanation, source)


def _peers(a: Author, pool: list[Author]) -> list[Author]:
    """Distractor authors from the same era group (classical vs modern), so the era alone gives nothing away."""
    return [b for b in pool if b.name != a.name and b.modern == a.modern]


def q_work_author(a: Author, rng: random.Random, canon=CANON) -> Question | None:
    owners = authors_of(canon)
    works = [w for w in a.works if len(owners[w]) == 1]
    if not works:
        return None
    w = rng.choice(works)
    peers = [b.name for b in _peers(a, list(canon)) if b.verse == a.verse or rng.random() < 0.3]
    if len(peers) < 3:
        return None
    return _mcq("work_author", f"«{w}» اثر کیست؟", a.name, rng.sample(peers, 3), rng, f"«{w}» از {a.name} است.")


def q_work_of(a: Author, rng: random.Random, canon=CANON) -> Question | None:
    if not a.works:
        return None
    own = set(a.works)
    others = [(w, b) for b in _peers(a, list(canon)) for w in b.works if w not in own]
    if len(others) < 3:
        return None
    picks = rng.sample(others, 3)
    if len({w for w, _ in picks}) < 3:
        return None
    w = rng.choice(a.works)
    note = "؛ ".join(f"«{x}» از {b.name}" for x, b in picks)
    return _mcq("work_of", f"کدام اثر از {a.name} است؟", w, [x for x, _ in picks], rng, f"«{w}» از {a.name} است ({note}).")


def q_not_by(a: Author, rng: random.Random, canon=CANON) -> Question | None:
    if len(a.works) < 3:
        return None
    own = set(a.works)
    others = [(w, b) for b in _peers(a, list(canon)) for w in b.works if w not in own]
    if not others:
        return None
    odd, owner = rng.choice(others)
    return _mcq("not_by", f"کدام اثر از {a.name} نیست؟", odd, rng.sample(a.works, 3), rng,
                f"«{odd}» از {owner.name} است؛ سه اثر دیگر از {a.name}‌اند.")


def q_century(a: Author, rng: random.Random, canon=CANON) -> Question | None:
    if a.death is None or a.modern:
        return None
    c = century(a.death)
    near = [x for x in range(c - 3, c + 4) if x != c and x in ORDINALS]
    if len(near) < 3:
        return None
    wrong = rng.sample(near, 3)
    return _mcq("century", f"{a.name} در کدام قرن (هجری قمری) درگذشت؟", century_label(c),
                [century_label(x) for x in wrong], rng, f"{a.name} حدود سال {a.death} هجری قمری درگذشت ({century_label(c)}).")


def q_style(a: Author, rng: random.Random, canon=CANON) -> Question | None:
    if a.style is None:
        return None
    wrong = rng.sample([s for s in STYLES if s != a.style], 3)
    who = "شاعران" if a.verse else "نویسندگان"
    return _mcq("style", f"{a.name} از {who} کدام سبک/دوره است؟", a.style, wrong, rng, f"{a.name}: {a.style}.")


def q_title(a: Author, rng: random.Random, canon=CANON) -> Question | None:
    if not a.title:
        return None
    peers = [b.name for b in _peers(a, list(canon)) if not b.title]
    return _mcq("title", f"لقب «{a.title}» از آنِ کیست؟", a.name, rng.sample(peers, 3), rng,
                f"«{a.title}» لقب {a.name} است.")


def q_chronology(rng: random.Random, canon=CANON, gap: int = 60) -> Question | None:
    """Four authors whose deaths are at least ``gap`` years apart (approximate dates stay unambiguous)."""
    dated = [a for a in canon if a.death is not None and not a.modern]
    rng.shuffle(dated)
    picks: list[Author] = []
    for a in dated:
        if all(abs(a.death - b.death) >= gap for b in picks):
            picks.append(a)
        if len(picks) == 4:
            break
    if len(picks) < 4:
        return None
    first = min(picks, key=lambda a: a.death)
    order = "، ".join(f"{a.name} ({a.death} ق)" for a in sorted(picks, key=lambda a: a.death))
    return _mcq("chronology", "کدام‌یک زودتر از بقیه درگذشته است؟", first.name,
                [a.name for a in picks if a is not first], rng, f"ترتیب درگذشت: {order}.")


_PERSIAN_MARKERS = {"که", "از", "به", "در", "را", "این", "آن", "با", "بر", "چو", "چون", "من", "تو", "است", "نیست",
                    "هر", "ز", "گر", "می", "ما", "دل", "نه", "تا"}


def _persian(verse: list[str]) -> bool:
    """Skip Arabic (ملمع) and Turkish (حیدربابا) couplets: a Persian couplet has Persian function words."""
    return sum(t in _PERSIAN_MARKERS for t in " ".join(verse).split()) >= 2


META = Path("data/raw/ganjoor_meta")
_DOUBTFUL = re.compile(r"منسوب|نامعلوم|ملحقات|مشکوک|منتسب|الحاقی|انتساب")


def fetch_meta(canon=CANON, meta: Path = META, threads: int = 8) -> None:
    """Cache poet.json and the first two category levels of every canon poet (for check / exclusions)."""
    from concurrent.futures import ThreadPoolExecutor

    from myllm.aruz.ganjoor import _fetch_json

    (meta / "poets").mkdir(parents=True, exist_ok=True)
    (meta / "cats").mkdir(parents=True, exist_ok=True)

    def one(slug: str) -> None:
        pf, cf = meta / "poets" / f"{slug}.json", meta / "cats" / f"{slug}.json"
        if not pf.exists():
            pf.write_text(json.dumps(_fetch_json(f"poets/{slug}/poet.json"), ensure_ascii=False), encoding="utf-8")
        if not cf.exists():
            root = _fetch_json(f"poets/{slug}/_cat.json")
            cats = []
            for child in root.get("ChildCats", []):
                cats.append({"title": child["Title"], "url": child["FullUrl"]})
                sub = _fetch_json(f"poets{child['FullUrl']}/_cat.json")
                cats += [{"title": g["Title"], "url": g["FullUrl"]} for g in sub.get("ChildCats", [])]
            cf.write_text(json.dumps(cats, ensure_ascii=False), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(one, sorted({a.slug for a in canon if a.slug})))


def doubtful_prefixes(slug: str, meta: Path = META) -> list[str]:
    """Category urls Ganjoor itself marks as attributed / unknown author (e.g. خواجو's «سام‌نامه»)."""
    f = meta / "cats" / f"{slug}.json"
    if not f.exists():
        return []
    return [c["url"].rstrip("/") + "/" for c in json.loads(f.read_text(encoding="utf-8")) if _DOUBTFUL.search(c["title"])]


def _couplets(slug: str, roots=ROOTS, exclude: list[str] | None = None) -> list[tuple[str, list[str]]]:
    out = []
    exclude = doubtful_prefixes(slug) if exclude is None else exclude
    for root in roots:
        d = Path(root) / slug
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.json")):
            poem = json.loads(f.read_text(encoding="utf-8"))
            if any(poem.get("FullUrl", "").startswith(p) for p in exclude):
                continue
            byc: dict[int, dict] = {}
            for v in poem.get("Verses", []):
                if v.get("Position") in ("Right", "Left"):
                    byc.setdefault(v["CoupletIndex"], {})[v["Position"]] = v["Text"]
            for c in sorted(byc)[:2]:  # the opening couplets are the best known
                if len(byc[c]) == 2 and _persian([byc[c]["Right"], byc[c]["Left"]]):
                    out.append((poem.get("FullUrl", ""), [byc[c]["Right"], byc[c]["Left"]]))
    return out


def q_verse_poet(a: Author, rng: random.Random, canon=CANON, cache: dict | None = None) -> Question | None:
    if not a.slug or not a.verse:
        return None
    cache = {} if cache is None else cache
    if a.slug not in cache:
        cache[a.slug] = _couplets(a.slug)
    pens = {p for b in canon for p in b.pen} | {b.name for b in canon}
    pool = [(u, v) for u, v in cache[a.slug] if not any(p in " ".join(v) for p in pens)]
    if not pool:
        return None
    url, verse = rng.choice(pool)
    peers = [b for b in _peers(a, list(canon)) if b.verse and b.slug]
    same = [b.name for b in peers if b.style and b.style == a.style]
    wrong = rng.sample(same, 1) if same else []
    rest = [b.name for b in peers if b.name not in wrong]
    wrong += rng.sample(rest, 3 - len(wrong))
    era = f"، {century_label(century(a.death))}" if a.death and not a.modern else ""
    style = f"، {a.style}" if a.style else ""
    return _mcq("verse_poet", "این بیت از کیست؟", a.name, wrong, rng, f"از {a.name}{style}{era}.", verse,
                f"https://ganjoor.net{url}")


KINDS = {"work_author": q_work_author, "work_of": q_work_of, "not_by": q_not_by, "century": q_century,
         "style": q_style, "title": q_title, "verse_poet": q_verse_poet}


def build(n: int = 400, seed: int = 42, kinds: set[str] | None = None, canon=CANON) -> list[Question]:
    rng = random.Random(seed)
    kinds = kinds or set(KINDS) | {"chronology"}
    cache: dict = {}
    bank, seen = [], set()
    for _ in range(n * 20):
        if len(bank) >= n:
            break
        kind = rng.choice(sorted(kinds))
        if kind == "chronology":
            q = q_chronology(rng, canon)
        elif kind == "verse_poet":
            q = q_verse_poet(rng.choice(canon), rng, canon, cache)
        else:
            q = KINDS[kind](rng.choice(canon), rng, canon)
        if q is None:
            continue
        key = (q.kind, q.prompt, tuple(q.verse)) + ((tuple(sorted(q.options)),) if q.kind == "chronology" else ())
        if key not in seen:
            seen.add(key)
            bank.append(q)
    return bank


def check(meta_dir: str = "data/raw/ganjoor_meta/poets", canon=CANON) -> tuple[list[str], dict[str, int]]:
    """Compare canon death centuries with Ganjoor's poet.json (cached under meta_dir).

    Returns (disagreements, counts). Ganjoor marks many dates as not validated; those still count as an
    independent second source, reported separately."""
    from myllm.aruz.ganjoor import _fetch_json

    Path(meta_dir).mkdir(parents=True, exist_ok=True)
    issues, counts = [], {"agree_validated": 0, "agree_unvalidated": 0, "disagree": 0, "missing": 0}
    for a in canon:
        if not a.slug or a.death is None:
            continue
        f = Path(meta_dir) / f"{a.slug}.json"
        if not f.exists():
            f.write_text(json.dumps(_fetch_json(f"poets/{a.slug}/poet.json"), ensure_ascii=False), encoding="utf-8")
        p = json.loads(f.read_text(encoding="utf-8"))
        g = p.get("DeathYearInLHijri") or None
        if not g:
            counts["missing"] += 1
        elif century(g) != century(a.death):
            counts["disagree"] += 1
            issues.append(f"{a.name}: canon {a.death} ({century(a.death)}) vs Ganjoor {g} ({century(g)})")
        else:
            counts["agree_validated" if p.get("ValidDeathDate") else "agree_unvalidated"] += 1
    return issues, counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=400)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--kinds", default="")
    b.add_argument("--out", default="data/processed/history_bank.jsonl")
    b.add_argument("--md", default="out/adab/history_worksheet.md")
    sub.add_parser("check")
    sub.add_parser("fetch")
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        fetch_meta()
        n = sum(len(doubtful_prefixes(x.slug)) for x in CANON if x.slug)
        print(f"cached metadata for canon poets in {META}; {n} attributed/unknown-author sections excluded")
        return 0
    if a.cmd == "check":
        issues, counts = check()
        print("\n".join(issues) or "no century disagreement with Ganjoor", counts)
        return 0
    from myllm.aruz.exercises import to_markdown

    bank = build(a.n, a.seed, set(a.kinds.split(",")) if a.kinds else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for q in bank:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text(to_markdown(bank), encoding="utf-8")
    from collections import Counter

    print(f"{len(bank)} questions -> {a.out} ({dict(Counter(q.kind for q in bank))}); worksheet -> {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
