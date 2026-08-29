"""Quality-tier normalizer tests — runnable standalone:

    python tests/test_normalizer_quality.py

The fast tier is pinned by test_normalizer.py; this file pins what the
quality tier adds: correct numbers in nine languages, the shapes the fast
tier has no rules for, and — most of all — the PASS ORDER, which is where
every bug in this module has come from so far.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from normalizer_quality import normalize  # noqa: E402

Q = lambda t, l: normalize(t, l, mode="quality")  # noqa: E731
F = lambda t, l: normalize(t, l, mode="fast")     # noqa: E731


def test_non_german_english_numbers_are_not_english():
    """The fast tier's number core only speaks de/en, so it reads Spanish
    numbers with English words. That is the whole reason this tier exists."""
    es = Q("Costó 1234 euros.", "es")
    assert "mil doscientos treinta y cuatro" in es, es
    assert "thousand" not in es
    fr = Q("Il a coûté 1234 euros.", "fr")
    assert "mille deux cent trente-quatre" in fr, fr
    ru = Q("Это стоило 1234 рубля.", "ru")
    assert "тысяча" in ru, ru
    it = Q("Costava 1234 euro.", "it")
    assert "milleduecentotrentaquattro" in it, it


def test_times_are_not_scores():
    """Pass-order regression: with _p_score ahead of the time rule, '14:30'
    became 'fourteen to thirty'."""
    for lang, needle in (("de", "vierzehn Uhr dreißig"),
                         ("en", "fourteen thirty"),
                         ("it", "quattordici e trenta"),
                         ("fr", "quatorze heures trente")):
        out = Q("Um 14:30." if lang == "de" else "At 14:30.", lang)
        assert needle in out, (lang, out)


def test_scores_are_not_times():
    assert "drei zu eins" in Q("Endstand 3:1.", "de")
    assert "three to one" in Q("Final score 3:1.", "en")


def test_dates_survive_the_fraction_and_math_passes():
    """Another pass-order regression: '15/06/1920' became 'fifteen divided
    by six' once, and '06/15/1920' fell through into the integer catch-all."""
    de = Q("Am 15.06.1920 war es soweit.", "de")
    assert "fünfzehnten Juni" in de, de          # dative kept from the fast tier
    en = Q("It happened on 06/15/1920.", "en")
    assert "June fifteenth" in en, en
    assert "divided" not in en
    es = Q("El 15/06/1920 pasó.", "es")
    assert "quince junio" in es, es


def test_roman_numerals_need_a_cue_and_read_by_context():
    assert "Ludwig der Vierzehnte" in Q("Ludwig XIV kam.", "de")
    assert "Charles the Fourth" in Q("Charles IV arrived.", "en")
    # a chapter is cardinal, a monarch is ordinal
    assert "Kapitel vier" in Q("Kapitel IV beginnt.", "de")
    assert "chapter four" in Q("chapter IV begins.", "en").lower()
    # no cue word: left alone (MIX, DID, I are words)
    assert "MIX" in Q("The MIX was good.", "en")


def test_shapes_the_fast_tier_has_no_rule_for():
    assert "one half" in Q("Take 1/2 of it.", "en")
    assert "ein halb" in Q("Nimm 1/2 davon.", "de")
    assert "trois quarts" in Q("Il en reste 3/4.", "fr")
    assert "nineteen nineties" in Q("Back in the 1990s.", "en")
    assert "er Jahre" in Q("In den 1990er Jahren.", "de")
    assert "third" in Q("On the 3rd.", "en")
    assert "two hours thirty minutes" in Q("It took 2h 30min.", "en")
    v = Q("Install v2.10.3 now.", "en")
    assert "version two" in v, v


def test_ranges_read_as_ranges():
    assert "bis" in Q("Der Krieg 1914-1918.", "de")
    assert " to " in Q("The 1914-1918 war.", "en")
    assert " a " in Q("La guerra de 1914-1918.", "es")


def test_units_are_spoken_where_a_table_exists_and_left_alone_otherwise():
    assert "Kilometer" in Q("Es sind 12 km.", "de")
    assert "kilómetros" in Q("Son 12 km.", "es")
    assert "километров" in Q("Это 12 км.", "ru")
    # Korean is outside both tiers' number support: digits and unit are left
    # for the engine, which reads them natively. What must NEVER happen is
    # English number words appearing inside a Korean sentence.
    out = Q("12 km 입니다.", "ko")
    assert "km" in out and "12" in out, out
    assert "twelve" not in out, out


def test_unsupported_language_falls_back_to_fast():
    t = "12 km 입니다."
    assert Q(t, "ko") == F(t, "ko")


def test_quality_never_raises_and_is_deterministic():
    weird = ["", "   ", "…", "1/0", "99:99:99", "v", "0.0.0.0", "€", "1e9",
             "1'234'567.89", "IIII", "31.02.2020", "24:00", "1234567890123456789"]
    for w in weird:
        for lang in ("de", "en", "es", "ru"):
            a = Q(w, lang)
            b = Q(w, lang)
            assert isinstance(a, str) and a == b, (w, lang, a, b)


def test_prose_is_untouched():
    p = "Ein ganz normaler Satz ohne Zahlen."
    assert Q(p, "de") == p
    e = "A perfectly ordinary sentence with no numbers."
    assert Q(e, "en") == e


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
