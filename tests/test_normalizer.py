"""Normalizer regression tests — runnable standalone (no pytest needed):

    python tests/test_normalizer.py

Every rule that ships gets pinned here so future edits can't silently
regress spoken output.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from normalizer import (  # noqa: E402
    de_ordinal, en_ordinal, en_year, de_year, normalize, _parse_number,
    digits_spelled,
)


def test_years_de():
    assert normalize("Im Jahr 1920 war es kalt.", "de") == \
        "Im Jahr neunzehnhundertzwanzig war es kalt."
    assert "neunzehnhundertfünf" in normalize("Es geschah 1905.", "de")
    assert "neunzehnhundert " in normalize("Um 1900 herum.", "de")
    # from 2000 on the German cardinal IS the year form
    assert "zweitausendfünfundzwanzig" in normalize("Bis 2025 fertig.", "de")


def test_years_en():
    assert normalize("In 1920 it was cold.", "en") == \
        "In nineteen twenty it was cold."
    assert "nineteen oh five" in normalize("Born in 1905.", "en")
    assert "nineteen hundred" in normalize("Around 1900.", "en")
    assert "twenty twenty-five" in normalize("By 2025.", "en")
    assert "two thousand nine" in normalize("Back in 2009.", "en")


def test_decades():
    assert "neunzehnhundertzwanziger" in normalize("die 1920er Jahre", "de")
    assert "neunzehnhundertneunzigern" in normalize("in den 1990ern", "de")
    assert "nineteen nineties" in normalize("the 1990s", "en")
    assert "nineteen hundreds" in normalize("the 1900s", "en")


def test_year_ranges():
    assert normalize("Der Krieg 1914-1918 war lang.", "de") == \
        "Der Krieg neunzehnhundertvierzehn bis neunzehnhundertachtzehn war lang."
    assert "nineteen fourteen to nineteen eighteen" in \
        normalize("The war 1914–1918 was long.", "en")  # en dash


def test_dates_de():
    # dative after "am", nominative otherwise
    assert normalize("Es begann am 15.06.1920.", "de") == \
        "Es begann am fünfzehnten Juni neunzehnhundertzwanzig."
    assert normalize("15.06.1920 war ein Dienstag.", "de") == \
        "fünfzehnter Juni neunzehnhundertzwanzig war ein Dienstag."
    assert "am ersten Januar" in normalize("am 01.01.2000", "de")
    assert "am dritten März" in normalize("am 3.3.1999", "de")
    assert "am einunddreißigsten Dezember" in normalize("am 31.12.1999", "de")


def test_dates_en():
    assert normalize("It began on 15/06/1920.", "en") == \
        "It began on June fifteenth, nineteen twenty."
    assert "January first" in normalize("on 01/01/2000", "en")
    assert "December twenty-first" in normalize("on 21/12/1980", "en")


def test_times():
    assert normalize("Es ist 14:30 Uhr.", "de") == "Es ist vierzehn Uhr dreißig."
    # Swiss/German dot notation must not become a decimal
    assert normalize("Treffen um 14.30 Uhr.", "de") == \
        "Treffen um vierzehn Uhr dreißig."
    assert normalize("Um 9.00 Uhr.", "de") == "Um neun Uhr."


def test_money():
    assert normalize("Das kostet CHF 12.50.", "de") == \
        "Das kostet zwölf Franken fünfzig."
    assert normalize("Das kostet Fr. 12.50.", "de") == \
        "Das kostet zwölf Franken fünfzig."
    assert "zwölf Franken" == normalize("CHF 12.00", "de")
    # regression: trailing sentence period must not join the amount
    assert normalize("Es kostet CHF 1'234.50.", "de") == \
        "Es kostet eintausendzweihundertvierunddreißig Franken fünfzig."
    assert normalize("It costs $3.99.", "en") == \
        "It costs three dollars ninety-nine."
    # 1-digit fraction stays a plain decimal
    assert "zwölf Komma fünf Franken" in normalize("12.5 CHF", "de")


def test_percent():
    assert normalize("Es sind 50% mehr.", "de") == "Es sind fünfzig Prozent mehr."
    assert "three point five percent" in normalize("about 3.5%", "en")


def test_units():
    assert "drei Komma fünf Kilogramm" in normalize("3.5 kg", "de")
    assert "fünfhundert Meter" in normalize("500 m weit", "de")
    assert "zwei Komma vier Gigahertz" in normalize("2.4 GHz", "de")
    assert "five hundred meters" in normalize("500 m away", "en")


def test_abbreviations():
    assert "das heißt" in normalize("d.h. morgen", "de")
    assert "Sankt Gallen" in normalize("St. Gallen", "de")
    assert "Saint Louis" in normalize("St. Louis", "en")
    assert "zum Beispiel" in normalize("z.B. hier", "de")


def test_abbreviations_need_word_boundary():
    # Confirmed bug, 2026-07-03: bare str.replace matched "min."/"ca."
    # *inside* other words instead of only at real abbreviation boundaries.
    assert normalize("Der Termin.", "de") == "Der Termin."
    assert normalize("Africa is big.", "en") == "Africa is big."
    assert normalize("Wir kommen in ca. 5 Minuten.", "de") == \
        "Wir kommen in circa fünf Minuten."


def test_ampersand():
    assert normalize("Klein & Fein", "de") == "Klein und Fein"
    assert normalize("Rock & Roll", "en") == "Rock and Roll"


def test_decimals_stay_decimals():
    assert "eintausendneunhundertzwanzig Komma fünf" in \
        normalize("Seite 1920.5", "de")
    assert "neunzehn Komma zwei" in normalize("19,2 Grad", "de")


def test_ordinal_helpers():
    assert en_ordinal(1) == "first"
    assert en_ordinal(12) == "twelfth"
    assert en_ordinal(20) == "twentieth"
    assert en_ordinal(21) == "twenty-first"
    assert en_ordinal(31) == "thirty-first"
    assert de_ordinal(1) == "erster"
    assert de_ordinal(7, "en") == "siebten"
    assert de_ordinal(20, "en") == "zwanzigsten"
    assert de_ordinal(31, "en") == "einunddreißigsten"


def test_year_helpers():
    assert en_year(1999) == "nineteen ninety-nine"
    assert en_year(2007) == "two thousand seven"
    assert de_year(1848) == "achtzehnhundertachtundvierzig"
    assert de_year(2030) == "zweitausenddreißig"


def test_prose_ordinals_de():
    # "N." + capitalized noun after a case-pinning article/preposition reads
    # as an ordinal with the right declension, not a bare cardinal. Was
    # "Das ist das eins. Mal." before 2026-07-05.
    assert normalize("Das ist das 1. Mal.", "de") == "Das ist das erste Mal."
    assert normalize("Der 3. Platz geht an Anna.", "de") == \
        "Der dritte Platz geht an Anna."
    assert normalize("Er wohnt im 3. Stock.", "de") == "Er wohnt im dritten Stock."
    assert normalize("Zum 1. Mal geklappt.", "de") == "Zum ersten Mal geklappt."
    assert normalize("Im 21. Jahrhundert.", "de") == "Im einundzwanzigsten Jahrhundert."
    assert normalize("Die 5. Klasse beginnt.", "de") == "Die fünfte Klasse beginnt."
    # dative feminine "der" — a preceding dative preposition selects "-en"
    assert normalize("Seit der 2. Woche geht es gut.", "de") == \
        "Seit der zweiten Woche geht es gut."
    assert normalize("In der 3. Klasse lernt man das.", "de") == \
        "In der dritten Klasse lernt man das."
    # day-ordinal + month-name dates (no year) come along for free
    assert normalize("Wir treffen uns am 15. Juni.", "de") == \
        "Wir treffen uns am fünfzehnten Juni."
    # must NOT fire: no case-pinning word before the number -> stays cardinal
    assert normalize("Die Antwort ist 5. Danach kommt mehr.", "de") == \
        "Die Antwort ist fünf. Danach kommt mehr."
    # must NOT fire: full numeric dates keep the dedicated date rule
    assert normalize("Es begann am 15.06.1920.", "de") == \
        "Es begann am fünfzehnten Juni neunzehnhundertzwanzig."
    # must NOT fire: dot-times keep the time rule
    assert normalize("Treffen um 14.30 Uhr.", "de") == \
        "Treffen um vierzehn Uhr dreißig."


def test_prose_ordinals_adversarial_de():
    # Findings from the 78-sentence adversarial grammar attack, 2026-07-05.
    # Finding 1: a capitalized FUNCTION word after "N." is a sentence start —
    # N is a cardinal there; rewriting merged sentences and changed meaning.
    assert normalize("Ich tippe auf die 3. Das ist meine Antwort.", "de") == \
        "Ich tippe auf die drei. Das ist meine Antwort."
    assert normalize("Er setzte alles auf die 7. Die Kugel rollte.", "de") == \
        "Er setzte alles auf die sieben. Die Kugel rollte."
    assert normalize("Der Kurs beginnt am 5. Bitte melden Sie sich an.", "de") == \
        "Der Kurs beginnt am fünf. Bitte melden Sie sich an."
    # Finding 2a: dative/genitive prepositions that were missing
    assert normalize("Vor der 1. Prüfung bin ich nervös.", "de") == \
        "Vor der ersten Prüfung bin ich nervös."
    assert normalize("Ab der 5. Klasse gibt es Französisch.", "de") == \
        "Ab der fünften Klasse gibt es Französisch."
    assert normalize("Innerhalb der 2. Woche kommt die Antwort.", "de") == \
        "Innerhalb der zweiten Woche kommt die Antwort."
    # Finding 2b: genitive "der" after a noun (sports/buildings/contests)
    assert normalize("Der Gewinner der 3. Runde steht fest.", "de") == \
        "Der Gewinner der dritten Runde steht fest."
    assert normalize("Kurz vor Ende der 1. Halbzeit fiel das Tor.", "de") == \
        "Kurz vor Ende der ersten Halbzeit fiel das Tor."
    # Finding 3: "-en"-plural nouns after "die"
    assert normalize("Die 3. Klassen gehen ins Museum.", "de") == \
        "Die dritten Klassen gehen ins Museum."
    # sentence-initial "Der" stays nominative
    assert normalize("Der 2. Versuch zählt.", "de") == "Der zweite Versuch zählt."


def test_urls_and_emails():
    assert normalize("Schreib an info@firma.de.", "de") == \
        "Schreib an info ät firma Punkt de."
    assert normalize("Email john.doe@gmail.com today.", "en") == \
        "Email john dot doe at gmail dot com today."
    assert normalize("Besuche www.example.ch.", "de") == \
        "Besuche www Punkt example Punkt ch."
    assert normalize("Go to https://example.com/docs now.", "en") == \
        "Go to example dot com slash docs now."
    # a trailing sentence period stays a sentence period, not "dot"
    out = normalize("Visit www.site.com.", "en")
    assert out.endswith("com."), out
    assert " dot com." in out, out
    # bare domains are deliberately left alone (false-positive risk)
    assert normalize("Der Punkt ist wichtig.", "de") == "Der Punkt ist wichtig."


def test_strips_markdown_and_junk():
    # Confirmed real, repeatedly-reported TTS failure mode: raw markdown
    # gets read as literal symbols ("asterisk asterisk"). 2026-07-05.
    assert normalize("This is **very** important and *also* nice.", "en") == \
        "This is very important and also nice."
    assert normalize("Check out [our site](https://example.com) for more.", "en") == \
        "Check out our site for more."
    assert normalize("Use `print(x)` to debug.", "en") == "Use print(x) to debug."
    assert "First point" in normalize("- First point\n- Second point", "en")
    assert normalize("# Welcome", "en") == "Welcome"
    assert normalize("> Quoted text here", "en") == "Quoted text here"
    assert normalize("---", "en") == ""
    # repeated punctuation collapses; runaway dots normalize to one ellipsis
    assert normalize("Amazing!!! Are you sure??", "en") == "Amazing! Are you sure?"
    assert normalize("Wait......", "en") == "Wait..."
    assert normalize("Wait. . .", "en") == "Wait..."
    # emoji stripped entirely
    assert "\U0001F600" not in normalize("Great job \U0001F600\U0001F389", "en")
    # smart quotes normalized to ASCII (so _parse_number's apostrophe rule
    # and other ASCII-pattern regexes still match pasted Word/web text)
    assert normalize("“Hello” she said.", "en") == '"Hello" she said.'
    # zero-width characters stripped so they can't silently break \b regexes
    assert normalize("z.​B. hier", "de") == "zum Beispiel hier"
    # numbered-list markers are deliberately NOT stripped (unlike bullets)
    # — "1. " collides with real German ordinal-in-prose constructions ("1.
    # Mal", "3. Platz"); the number must survive in some spoken form, not
    # be silently dropped as if it were a list marker. (The exact spoken
    # form — "eins." vs the grammatically correct "ersten" — is a separate,
    # already-flagged follow-up; this only pins that stripping doesn't eat it.)
    assert normalize("1. Mal geklappt", "de") != "Mal geklappt"


def test_parse_number():
    assert _parse_number("1'234.50", "de") == ("1234", "50")
    assert _parse_number("1.234", "de") == ("1234", "")
    assert _parse_number("3.5", "de") == ("3", "5")
    assert _parse_number("1,234.5", "en") == ("1234", "5")
    assert _parse_number("3,5", "de") == ("3", "5")


def test_leading_zero_identifiers():
    # order/reference/room numbers with a leading zero are identifiers, not
    # quantities — int() would otherwise silently drop the zero ("016" ->
    # "sechzehn"). Confirmed bug, 2026-07-05.
    assert normalize("Die Bestellung Nr. 016 ist da.", "de") == \
        "Die Bestellung Nummer null eins sechs ist da."
    assert normalize("Zimmer 007.", "de") == "Zimmer null null sieben."
    assert normalize("Order number 016 shipped.", "en") == \
        "Order number zero one six shipped."
    # phone-number-shaped groups: the leading-zero group reads digit-by-
    # digit, groups without a leading zero keep their normal cardinal
    # reading — matches DIN 5008's paired-group convention, not a blanket
    # digit-by-digit reading of the whole number.
    assert normalize("Ruf mich unter 016 34 56 an.", "de") == \
        "Ruf mich unter null eins sechs vierunddreißig sechsundfünfzig an."
    # a lone "0" is a real cardinal (e.g. a temperature, a score) and must
    # stay "null"/"zero", not fall into the identifier path (len(int_str)>1
    # excludes it).
    assert normalize("Es sind 0 Grad.", "de") == "Es sind null Grad."
    # unaffected paths: dates/times bypass _spell_amount entirely, so a
    # literal leading zero in "01.01.2000" or "09:05" must keep reading as
    # a normal ordinal/hour, not digit-by-digit.
    assert "am ersten Januar" in normalize("am 01.01.2000", "de")
    assert normalize("Es ist 09:05 Uhr.", "de") == "Es ist neun Uhr fünf."


def test_digits_spelled_helper():
    assert digits_spelled("016", "de") == "null eins sechs"
    assert digits_spelled("007", "en") == "zero zero seven"


def test_never_crashes():
    for weird in ("", "...", "1234567890123", "%%%&&&", "12.34.56.78.90",
                  "  ", "am 99.99.9999"):
        normalize(weird, "de")
        normalize(weird, "en")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
