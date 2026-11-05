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
