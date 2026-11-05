"""Adaptive tutor: a learner profile per user, weakest-skill-first practice, spaced review, report card.

    tutor = Tutor(path="out/aruz/profiles.json")      # or path=None: in memory only
    q = tutor.pick(user, bank)                         # next question for this learner
    tutor.record_answer(user, q, correct=True)
    tutor.record_check(user, diagnosis)                # from /check (myllm.aruz.diagnose)
    print(tutor.report(user))                          # /karname

* Skill choice: Thompson sampling on each skill's error rate, Beta(1 + wrong, 1 + right), so weak skills
  come up more often while new skills still get explored.
* Spaced review (Leitner): a wrong answer comes back after 3 questions, then after 6, 12, … until it is
  answered right twice in a row.
* Privacy: users are stored under a salted hash (ARUZ_PROFILE_SALT), never the raw chat id, and nothing
  but counts is kept (no message text).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from myllm.aruz.diagnose import RULES
from myllm.aruz.exercises import Question

SKILL_OF_KIND = {
    "meter": "meter", "taqti": "taqti", "ikhtiyar": "ikhtiyar", "rhyme": "rhyme",
    "work_author": "works", "work_of": "works", "not_by": "works", "century": "periods", "chronology": "periods",
    "style": "styles", "title": "people", "verse_poet": "verse_poet",
}
SKILLS = {
    "meter": ("تشخیص وزن", "اول تقطیع هجایی کن، بعد نشانه‌ها را با ارکان بحرها مقایسه کن."),
    "taqti": ("تقطیع هجایی", "با /check تقطیع خودت را بفرست تا هجای اشتباه و قاعده‌اش را ببینی."),
    "ikhtiyar": ("اختیارات شاعری", "حذف همزه، بلندخوانی، تسکین، «فاعلاتن» به‌جای «فعلاتن» و قلب را مرور کن."),
    "rhyme": ("قافیه و ردیف", "ردیف: واژه‌های تکراریِ پایان مصراع؛ قافیه: واژه‌ی هم‌آوای پیش از ردیف."),
    "works": ("آثار ادبی", "آثار هر شاعر را همراه سبک و قرنش به خاطر بسپار."),
    "periods": ("دوره‌ها و تاریخ", "هر قرن را با یک شاعر شاخص به یاد بسپار: رودکی ۴، فردوسی ۵، سعدی ۷، حافظ ۸، جامی ۹."),
    "styles": ("سبک‌شناسی", "خراسانی تا حدود قرن ۶، عراقی ۷ تا ۹، هندی ۱۰ تا ۱۲، سپس بازگشت، مشروطه و معاصر."),
    "people": ("شاعران و لقب‌ها", "لسان‌الغیب: حافظ؛ حکیم توس: فردوسی؛ حجت خراسان: ناصرخسرو؛ خاتم‌الشعرا: جامی."),
    "verse_poet": ("شناخت شاعر از بیت", "به واژگان، مضمون و قالب بیت دقت کن تا دوره و سبکش را حدس بزنی."),
}
_FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(x) -> str:
    return str(x).translate(_FA)


def qkey(q: Question) -> str:
    raw = json.dumps([q.kind, q.prompt, q.verse, sorted(q.options)], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Profile:
    stats: dict[str, list[int]] = field(default_factory=dict)  # skill -> [seen, correct]
    rules: dict[str, int] = field(default_factory=dict)  # /check rule id -> times missed
    syllables: list[int] = field(default_factory=lambda: [0, 0])  # /check: [total, correct]
    boxes: dict[str, int] = field(default_factory=dict)  # question key -> Leitner box
    due: dict[str, int] = field(default_factory=dict)  # question key -> turn when it should come back
    turn: int = 0


class Tutor:
    def __init__(self, seed: int = 0, path: str | Path | None = None, salt: str | None = None):
        self.rng = random.Random(seed)
        self.path = Path(path) if path else None
        self.salt = os.environ.get("ARUZ_PROFILE_SALT", "") if salt is None else salt
        self.profiles: dict[str, Profile] = {}
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.profiles = {k: Profile(**v) for k, v in raw.items()}

    # --- storage ----------------------------------------------------------------------------------
    def _id(self, user) -> str:
        return hashlib.sha256(f"{self.salt}:{user}".encode()).hexdigest()[:20]

    def profile(self, user) -> Profile:
        return self.profiles.setdefault(self._id(user), Profile())

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: asdict(v) for k, v in self.profiles.items()}, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self.path)

    # --- learning signal --------------------------------------------------------------------------
    def record_answer(self, user, q: Question, correct: bool) -> None:
        p = self.profile(user)
        skill = SKILL_OF_KIND.get(q.kind, q.kind)
        seen, right = p.stats.get(skill, [0, 0])
        p.stats[skill] = [seen + 1, right + int(correct)]
        k = qkey(q)
        if not correct:
            p.boxes[k] = 1
            p.due[k] = p.turn + 3
        elif k in p.boxes:  # a review question answered right: wait longer, retire after two in a row
            p.boxes[k] += 1
            if p.boxes[k] >= 3:
                p.due.pop(k, None)
            else:
                p.due[k] = p.turn + 3 * 2 ** p.boxes[k]
        self.save()

    def record_check(self, user, diagnosis) -> None:
        p = self.profile(user)
        p.syllables[0] += len(diagnosis.expected)
        p.syllables[1] += diagnosis.correct
        for m in diagnosis.mistakes:
            if m.rule:
                p.rules[m.rule] = p.rules.get(m.rule, 0) + 1
        seen, right = p.stats.get("taqti", [0, 0])
        p.stats["taqti"] = [seen + 1, right + int(not diagnosis.mistakes and bool(diagnosis.expected))]
        self.save()

    # --- choosing the next question ---------------------------------------------------------------
    def pick(self, user, bank: list[Question]) -> Question:
        p = self.profile(user)
        p.turn += 1
        index = {qkey(q): q for q in bank}
        due = sorted((t, k) for k, t in p.due.items() if t <= p.turn and k in index)
        if due:
            return index[due[0][1]]
        by_skill: dict[str, list[Question]] = {}
        for q in bank:
            by_skill.setdefault(SKILL_OF_KIND.get(q.kind, q.kind), []).append(q)

        def sampled_error(skill: str) -> float:
            seen, right = p.stats.get(skill, [0, 0])
            return self.rng.betavariate(1 + seen - right, 1 + right)

        skill = max(sorted(by_skill), key=sampled_error)
        fresh = [q for q in by_skill[skill] if qkey(q) not in p.boxes]
        return self.rng.choice(fresh or by_skill[skill])

    # --- report card ------------------------------------------------------------------------------
    def report(self, user) -> str:
        p = self.profile(user)
        if not p.stats and not p.syllables[0]:
            return "هنوز پاسخی ثبت نشده. با /tamrin (موسیقی شعر)، /tarikh (تاریخ ادبیات) یا /check شروع کن."
        seen = sum(s for s, _ in p.stats.values())
        right = sum(r for _, r in p.stats.values())
        lines = ["📊 کارنامه‌ی تو", f"پاسخ‌ها: {fa(seen)} — درست: {fa(right)} ({fa(round(100 * right / max(1, seen)))}٪)", ""]
        rows = []
        for skill, (s, r) in sorted(p.stats.items(), key=lambda kv: kv[1][1] / max(1, kv[1][0])):
            label = SKILLS.get(skill, (skill, ""))[0]
            pct = r / max(1, s)
            bar = "▓" * round(10 * pct) + "░" * (10 - round(10 * pct))
            rows.append((pct, s, skill))
            lines.append(f"• {label}: {fa(r)} از {fa(s)} {bar} {fa(round(100 * pct))}٪")
        weak = [(pct, skill) for pct, s, skill in rows if s >= 3 and pct < 0.8]
        if weak:
            pct, skill = min(weak)
            label, tip = SKILLS.get(skill, (skill, ""))
            lines += ["", f"🔻 بیشترین نیاز به تمرین: {label} ({fa(round(100 * pct))}٪) — {tip}"]
        if p.syllables[0]:
            acc = p.syllables[1] / p.syllables[0]
            lines += ["", f"✍️ تقطیع با /check: {fa(p.syllables[1])} از {fa(p.syllables[0])} نشانه درست ({fa(round(100 * acc))}٪)"]
        if p.rules:
            lines.append("🧩 قاعده‌هایی که بیشتر جا انداختی:")
            for rule, n in sorted(p.rules.items(), key=lambda kv: -kv[1])[:3]:
                name, tip = RULES.get(rule, (rule, ""))
                lines.append(f"• {name} — {fa(n)} بار: {tip}")
        waiting = len(p.due)
        if waiting:
            lines += ["", f"⏰ {fa(waiting)} سؤالِ اشتباه‌شده برای مرور در صف است؛ با /tamrin یا /tarikh برمی‌گردند."]
        return "\n".join(lines)
