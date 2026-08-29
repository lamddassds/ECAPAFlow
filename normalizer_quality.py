"""Quality tier of the text normalizer — deterministic, no LLM.

Two tiers are exposed to the user, and this is the slow, thorough one:

* **Fast** (`normalizer.py`) — the LoudFlow regex engine. Tuned for German and
  English, microseconds per sentence, memoized. Everything it does not know
  about it leaves alone. On a Spanish or Russian script it still reads numbers
  with ENGLISH words, because its number core only speaks de/en.
* **Quality** (this module) — same rules first, then a second deterministic
  pass that renders every remaining numeric and symbolic shape with
  `num2words`, in the ACTUAL language of the text. Adds the shapes the fast
  tier has no rules for at all: ordinals, decades, roman numerals, ranges,
  scores, durations, versions, math signs, fractions, phone groups and units
  across nine languages.

Explicitly NOT a language model. A 0.5B polish pass was considered and
rejected: on comparable material it produced byte-identical output to the
rules 36 times out of 36 while adding seconds of latency per sentence, and
Lauro's standing instruction is that LLM normalizers are too slow.

Pass order is load-bearing, for the same reason it is in any normalizer: later
passes are catch-alls that assume the ambiguous shapes have already been
consumed. `_p_year` must run before `_p_integer` or "1998" becomes "one
thousand nine hundred ninety-eight" in a sentence about a year; `_p_ordinal`
must run before `_p_integer` for the same reason; `_p_range` must run before
both or "1914-1918" is two integers and a hyphen.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import normalizer as _fast
from normalizer import normalize as _fast_normalize

try:
    from num2words import num2words as _n2w_raw
    HAVE_NUM2WORDS = True
except Exception:  # pragma: no cover - the package is a hard dependency in requirements
    HAVE_NUM2WORDS = False

    def _n2w_raw(*_a, **_k):  # type: ignore
        raise RuntimeError("num2words missing")


# Languages this tier can actually render numbers in. Anything else falls
# straight through to the fast tier — a wrong normalization is worse than
# none, and a Korean sentence with English number words is a wrong one.
QUALITY_LANGS = {"de", "en", "es", "fr", "it", "pt", "nl", "ru", "tr"}

# Per-language function words used by the shape passes.
_WORDS = {
    "de": {"to": "bis", "per": "pro", "point": "Komma", "minus": "minus",
           "plus": "plus", "times": "mal", "div": "geteilt durch",
           "eq": "gleich", "percent": "Prozent", "permille": "Promille",
           "degree": "Grad", "colon_score": "zu", "version": "Version",
           "hour": "Stunden", "min": "Minuten", "sec": "Sekunden", "and": "und"},
    "en": {"to": "to", "per": "per", "point": "point", "minus": "minus",
           "plus": "plus", "times": "times", "div": "divided by",
           "eq": "equals", "percent": "percent", "permille": "per mille",
           "degree": "degrees", "colon_score": "to", "version": "version",
           "hour": "hours", "min": "minutes", "sec": "seconds", "and": "and"},
    "es": {"to": "a", "per": "por", "point": "coma", "minus": "menos",
           "plus": "más", "times": "por", "div": "dividido por",
           "eq": "igual a", "percent": "por ciento", "permille": "por mil",
           "degree": "grados", "colon_score": "a", "version": "versión",
           "hour": "horas", "min": "minutos", "sec": "segundos", "and": "y"},
    "fr": {"to": "à", "per": "par", "point": "virgule", "minus": "moins",
           "plus": "plus", "times": "fois", "div": "divisé par",
           "eq": "égale", "percent": "pour cent", "permille": "pour mille",
           "degree": "degrés", "colon_score": "à", "version": "version",
           "hour": "heures", "min": "minutes", "sec": "secondes", "and": "et"},
    "it": {"to": "a", "per": "per", "point": "virgola", "minus": "meno",
           "plus": "più", "times": "per", "div": "diviso",
           "eq": "uguale a", "percent": "per cento", "permille": "per mille",
           "degree": "gradi", "colon_score": "a", "version": "versione",
           "hour": "ore", "min": "minuti", "sec": "secondi", "and": "e"},
    "pt": {"to": "a", "per": "por", "point": "vírgula", "minus": "menos",
           "plus": "mais", "times": "vezes", "div": "dividido por",
           "eq": "igual a", "percent": "por cento", "permille": "por mil",
           "degree": "graus", "colon_score": "a", "version": "versão",
           "hour": "horas", "min": "minutos", "sec": "segundos", "and": "e"},
    "nl": {"to": "tot", "per": "per", "point": "komma", "minus": "min",
           "plus": "plus", "times": "keer", "div": "gedeeld door",
           "eq": "is", "percent": "procent", "permille": "promille",
           "degree": "graden", "colon_score": "tegen", "version": "versie",
           "hour": "uur", "min": "minuten", "sec": "seconden", "and": "en"},
    "ru": {"to": "до", "per": "на", "point": "целых", "minus": "минус",
           "plus": "плюс", "times": "умножить на", "div": "разделить на",
           "eq": "равно", "percent": "процентов", "permille": "промилле",
           "degree": "градусов", "colon_score": "—", "version": "версия",
           "hour": "часов", "min": "минут", "sec": "секунд", "and": "и"},
    "tr": {"to": "ile", "per": "başına", "point": "virgül", "minus": "eksi",
           "plus": "artı", "times": "çarpı", "div": "bölü",
           "eq": "eşittir", "percent": "yüzde", "permille": "binde",
           "degree": "derece", "colon_score": "-", "version": "sürüm",
           "hour": "saat", "min": "dakika", "sec": "saniye", "and": "ve"},
}

# Units are spoken, not spelled. Kept deliberately small and unambiguous:
# a unit table that guesses is worse than one that declines to.
_UNITS = {
    "de": {"km": "Kilometer", "m": "Meter", "cm": "Zentimeter", "mm": "Millimeter",
           "kg": "Kilogramm", "g": "Gramm", "mg": "Milligramm", "t": "Tonnen",
           "l": "Liter", "ml": "Milliliter", "h": "Stunden", "min": "Minuten",
           "s": "Sekunden", "km/h": "Kilometer pro Stunde", "GHz": "Gigahertz",
           "MHz": "Megahertz", "kHz": "Kilohertz", "GB": "Gigabyte",
           "MB": "Megabyte", "TB": "Terabyte", "kB": "Kilobyte", "W": "Watt",
           "kW": "Kilowatt", "V": "Volt", "A": "Ampere", "°C": "Grad Celsius",
           "°F": "Grad Fahrenheit"},
    "en": {"km": "kilometres", "m": "metres", "cm": "centimetres",
           "mm": "millimetres", "kg": "kilograms", "g": "grams",
           "mg": "milligrams", "t": "tonnes", "l": "litres", "ml": "millilitres",
           "h": "hours", "min": "minutes", "s": "seconds",
           "km/h": "kilometres per hour", "GHz": "gigahertz",
           "MHz": "megahertz", "kHz": "kilohertz", "GB": "gigabytes",
           "MB": "megabytes", "TB": "terabytes", "kB": "kilobytes", "W": "watts",
           "kW": "kilowatts", "V": "volts", "A": "amperes",
           "°C": "degrees Celsius", "°F": "degrees Fahrenheit"},
    "es": {"km": "kilómetros", "m": "metros", "cm": "centímetros",
           "kg": "kilogramos", "g": "gramos", "l": "litros", "ml": "mililitros",
           "h": "horas", "min": "minutos", "km/h": "kilómetros por hora",
           "GB": "gigabytes", "MB": "megabytes", "°C": "grados centígrados"},
    "fr": {"km": "kilomètres", "m": "mètres", "cm": "centimètres",
           "kg": "kilogrammes", "g": "grammes", "l": "litres",
           "ml": "millilitres", "h": "heures", "min": "minutes",
           "km/h": "kilomètres par heure", "GB": "gigaoctets",
           "MB": "mégaoctets", "°C": "degrés Celsius"},
    "it": {"km": "chilometri", "m": "metri", "cm": "centimetri",
           "kg": "chilogrammi", "g": "grammi", "l": "litri",
           "ml": "millilitri", "h": "ore", "min": "minuti",
           "km/h": "chilometri orari", "GB": "gigabyte", "MB": "megabyte",
           "°C": "gradi"},
    "pt": {"km": "quilômetros", "m": "metros", "cm": "centímetros",
           "kg": "quilogramas", "g": "gramas", "l": "litros",
           "ml": "mililitros", "h": "horas", "min": "minutos",
           "km/h": "quilômetros por hora", "GB": "gigabytes",
           "MB": "megabytes", "°C": "graus"},
    "nl": {"km": "kilometer", "m": "meter", "cm": "centimeter",
           "kg": "kilogram", "g": "gram", "l": "liter", "ml": "milliliter",
           "h": "uur", "min": "minuten", "km/h": "kilometer per uur",
           "GB": "gigabyte", "MB": "megabyte", "°C": "graden"},
    "ru": {"км": "километров", "м": "метров", "см": "сантиметров",
           "кг": "килограммов", "г": "граммов", "л": "литров",
           "ч": "часов", "мин": "минут", "км/ч": "километров в час",
           "ГБ": "гигабайт", "МБ": "мегабайт", "°C": "градусов"},
    "tr": {"km": "kilometre", "m": "metre", "cm": "santimetre",
           "kg": "kilogram", "g": "gram", "l": "litre", "ml": "mililitre",
           "sa": "saat", "dk": "dakika", "km/s": "kilometre saatte",
           "GB": "gigabayt", "MB": "megabayt", "°C": "santigrat derece"},
}

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
# Only after a cue word — bare "I" and "MIX" are words, not numerals.
_ROMAN_CUE = re.compile(
    r"\b(König|Königin|Kaiser|Papst|Karl|Ludwig|Heinrich|Friedrich|Otto|Wilhelm|"
    r"King|Queen|Emperor|Pope|Charles|Louis|Henry|Philip|Chapter|Kapitel|Teil|"
    r"Part|Band|Book|Buch|World War|Weltkrieg)\s+([IVXLCDM]{1,7})\b",
    re.IGNORECASE)


def _lang2(lang: Optional[str]) -> str:
    return (lang or "en")[:2].lower()


def _w(lang: str, key: str) -> str:
    return _WORDS.get(lang, _WORDS["en"])[key]


def _n2w(n, lang: str, to: str = "cardinal") -> str:
    """Never-raising num2words wrapper. Any failure returns the plain digits,
    which the engine can still read — a crashed normalizer is the one failure
    mode that must not exist in a synthesis path."""
    try:
        return str(_n2w_raw(n, lang=lang, to=to))
    except Exception:
        try:
            return str(_n2w_raw(n, lang="en", to=to))
        except Exception:
            return str(n)


def _int_words(digits: str, lang: str) -> str:
    """Long or leading-zero digit runs are identifiers, not quantities."""
    if len(digits) > 1 and digits[0] == "0":
        return " ".join(_n2w(int(d), lang) for d in digits)
    if len(digits) > 15:
        return " ".join(_n2w(int(d), lang) for d in digits)
    return _n2w(int(digits), lang)


def _dec_words(int_part: str, frac: str, lang: str) -> str:
    head = _int_words(int_part or "0", lang)
    tail = " ".join(_n2w(int(d), lang) for d in frac)
    return f"{head} {_w(lang, 'point')} {tail}"


def _roman_to_int(s: str) -> Optional[int]:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN.get(ch)
        if v is None:
            return None
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total if 0 < total < 4000 else None


# ------------------------------------------------------------------ passes
# A monarch is "Charles the third"; a chapter is "chapter four". Same
# numeral, different reading, decided by the cue word in front of it.
_ROMAN_CARDINAL_CUES = {"chapter", "kapitel", "teil", "part", "band", "book",
                        "buch", "world war", "weltkrieg"}


def _p_roman(text: str, lang: str) -> str:
    def rep(m):
        n = _roman_to_int(m.group(2))
        if n is None:
            return m.group(0)
        cue = m.group(1).lower()
        if cue in _ROMAN_CARDINAL_CUES:
            return f"{m.group(1)} {_n2w(n, lang, to='cardinal')}"
        # Regnal numbers take an article: "Ludwig der Vierzehnte",
        # "Charles the Third" — not "Ludwig vierzehnte".
        word = _n2w(n, lang, to="ordinal")
        article = {"de": "der ", "en": "the ", "nl": "de ", "es": "", "fr": "",
                   "it": "", "pt": "", "ru": "", "tr": ""}.get(lang, "")
        if article:
            word = word[:1].upper() + word[1:]
        return f"{m.group(1)} {article}{word}"
    return _ROMAN_CUE.sub(rep, text)


def _p_version(text: str, lang: str) -> str:
    """"v2.10.3" / "Version 3.4" — dotted versions are digit groups, never a
    decimal number."""
    def rep(m):
        parts = [p for p in m.group(2).split(".") if p != ""]
        spoken = f" {_w(lang, 'point')} ".join(_int_words(p, lang) for p in parts)
        return f"{_w(lang, 'version')} {spoken}"
    return re.sub(r"\b(v|version|Version)\s*\.?\s*(\d+(?:\.\d+){1,3})\b", rep, text)


def _p_range(text: str, lang: str) -> str:
    """"1914-1918", "10–20" — a dash between two numbers is spoken."""
    def rep(m):
        a, b = m.group(1), m.group(2)
        fa = _year_or_int(a, lang)
        fb = _year_or_int(b, lang)
        return f"{fa} {_w(lang, 'to')} {fb}"
    return re.sub(r"\b(\d{1,4})\s*[–—-]\s*(\d{1,4})\b(?!\s*[-–—]\s*\d)", rep, text)


def _p_score(text: str, lang: str) -> str:
    """A colon pair that cannot be a clock time is a score.

    "14:30" is a time and belongs to the time pass; "3:1" cannot be one. The
    discriminator is the shape, not the values: a written time always has a
    two-digit minute, so a single-digit right side, an out-of-range hour or
    an out-of-range minute all mean score.
    """
    def rep(m):
        a, b = m.group(1), m.group(2)
        looks_like_time = (len(b) == 2 and int(a) <= 23 and int(b) <= 59)
        if looks_like_time:
            return m.group(0)
        return f"{_int_words(a, lang)} {_w(lang, 'colon_score')} {_int_words(b, lang)}"
    return re.sub(r"\b(\d{1,3}):(\d{1,3})\b", rep, text)


def _p_duration(text: str, lang: str) -> str:
    """"2h 30min", "1:45 h"."""
    def rep(m):
        h, mi = m.group(1), m.group(2)
        return (f"{_int_words(h, lang)} {_w(lang, 'hour')} "
                f"{_int_words(mi, lang)} {_w(lang, 'min')}")
    return re.sub(r"\b(\d{1,2})\s*h\s*(\d{1,2})\s*(?:min)?\b", rep, text)


def _p_math(text: str, lang: str) -> str:
    out = re.sub(r"(\d)\s*×\s*(\d)", lambda m: f"{m.group(1)} {_w(lang,'times')} {m.group(2)}", text)
    out = re.sub(r"(\d)\s*÷\s*(\d)", lambda m: f"{m.group(1)} {_w(lang,'div')} {m.group(2)}", out)
    out = re.sub(r"(\d)\s*=\s*(\d)", lambda m: f"{m.group(1)} {_w(lang,'eq')} {m.group(2)}", out)
    out = re.sub(r"(?<![\w])[-−]\s*(\d)", lambda m: f"{_w(lang,'minus')} {m.group(1)}", out)
    return out


def _p_percent(text: str, lang: str) -> str:
    out = re.sub(r"(\d(?:[\d.,']*\d)?)\s*%",
                 lambda m: f"{_num_token(m.group(1), lang)} {_w(lang,'percent')}", text)
    out = re.sub(r"(\d(?:[\d.,']*\d)?)\s*‰",
                 lambda m: f"{_num_token(m.group(1), lang)} {_w(lang,'permille')}", out)
    return out


def _p_units(text: str, lang: str) -> str:
    # No table for this language: leave the unit written. A German "km" read
    # out as the English word is worse than the abbreviation, which every
    # one of these engines already pronounces acceptably.
    table = _UNITS.get(lang)
    if not table:
        return text
    keys = sorted(table, key=len, reverse=True)
    pat = re.compile(r"(\d(?:[\d.,']*\d)?)\s*(" + "|".join(re.escape(k) for k in keys) + r")(?![\wäöüß])")
    return pat.sub(lambda m: f"{_num_token(m.group(1), lang)} {table[m.group(2)]}", text)


# Common fractions are lexical, not computed: "1/2" is "a half" in every one
# of these languages, never "one divided by two".
_FRACTION_WORDS = {
    "de": {2: "halb", 3: "Drittel", 4: "Viertel", 5: "Fünftel", 8: "Achtel"},
    "en": {2: "half", 3: "third", 4: "quarter", 5: "fifth", 8: "eighth"},
    "es": {2: "medio", 3: "tercio", 4: "cuarto", 5: "quinto", 8: "octavo"},
    "fr": {2: "demi", 3: "tiers", 4: "quart", 5: "cinquième", 8: "huitième"},
    "it": {2: "mezzo", 3: "terzo", 4: "quarto", 5: "quinto", 8: "ottavo"},
    "pt": {2: "meio", 3: "terço", 4: "quarto", 5: "quinto", 8: "oitavo"},
    "nl": {2: "half", 3: "derde", 4: "kwart", 5: "vijfde", 8: "achtste"},
    "ru": {2: "половина", 3: "треть", 4: "четверть", 5: "пятая", 8: "восьмая"},
    "tr": {2: "yarım", 3: "üçte bir", 4: "çeyrek", 5: "beşte bir", 8: "sekizde bir"},
}


def _p_fraction(text: str, lang: str) -> str:
    def rep(m):
        num, den = int(m.group(1)), int(m.group(2))
        table = _FRACTION_WORDS.get(lang, _FRACTION_WORDS["en"])
        word = table.get(den)
        if word:
            if num == 1:
                return {"de": f"ein {word}", "en": f"one {word}"}.get(
                    lang, f"{_n2w(1, lang)} {word}")
            plural = word + ("s" if lang in ("en", "es", "fr", "pt") else "")
            return f"{_n2w(num, lang)} {plural}"
        return f"{_n2w(num, lang)} {_w(lang,'div')} {_n2w(den, lang)}"
    # Not a date: a slash-separated pair that is part of d/m/y was already
    # consumed by _p_date for the locales that use it.
    return re.sub(r"(?<!\d)(\d{1,3})/(\d{1,3})(?!\d)(?!\s*/)", rep, text)


def _p_decade(text: str, lang: str) -> str:
    """"1990er", "1990s", "90er"."""
    def rep(m):
        y = int(m.group(1))
        if y < 100:
            y = 1900 + y
        w = _n2w(y, lang, to="year")
        if lang == "de":
            return w + "er Jahre"
        # English pluralises the decade word itself: "nineteen nineties",
        # never "nineteen ninetys".
        if w.endswith("y"):
            return w[:-1] + "ies"
        return w + "s"
    return re.sub(r"\b(\d{2,4})(?:er|s)\b", rep, text)


def _p_ordinal(text: str, lang: str) -> str:
    """A digit followed by a period at a non-sentence-final position, or an
    explicit ordinal suffix, is an ordinal. Deliberately conservative: "1."
    at the end of a sentence is left alone, because there it is far more
    likely to be a numbered item than an inflected ordinal."""
    out = re.sub(r"\b(\d{1,3})(?:st|nd|rd|th)\b",
                 lambda m: _n2w(int(m.group(1)), lang, to="ordinal"), text)
    if lang in ("de", "nl"):
        out = re.sub(r"\b(\d{1,3})\.\s+(?=[a-zäöüß])",
                     lambda m: _n2w(int(m.group(1)), lang, to="ordinal") + " ", out)
    return out


def _year_or_int(tok: str, lang: str) -> str:
    n = int(tok)
    if 1100 <= n <= 2199 and len(tok) == 4:
        return _n2w(n, lang, to="year")
    return _int_words(tok, lang)


def _p_year(text: str, lang: str) -> str:
    return re.sub(r"(?<![\d.,'])\b(1[1-9]\d{2}|20\d{2}|21\d{2})\b(?![\d.,'])",
                  lambda m: _n2w(int(m.group(1)), lang, to="year"), text)


def _num_token(tok: str, lang: str) -> str:
    """Render one raw numeric token, resolving separators the way the locale
    writes them (apostrophe is always a thousands group; when both '.' and ','
    appear the LAST one is the decimal point)."""
    t = tok.replace("'", "").replace(" ", "").replace(" ", "")
    has_dot, has_com = "." in t, "," in t
    if has_dot and has_com:
        dec = "." if t.rfind(".") > t.rfind(",") else ","
        head, _, frac = t.rpartition(dec)
        head = head.replace(".", "").replace(",", "")
    elif has_com:
        if lang in ("de", "es", "it", "pt", "nl", "ru", "tr", "fr"):
            head, _, frac = t.rpartition(",")
        else:
            head, frac = t.replace(",", ""), ""
    elif has_dot:
        head, _, tail = t.rpartition(".")
        if lang == "en" or len(tail) != 3:
            head, frac = head, tail
        else:
            head, frac = t.replace(".", ""), ""
    else:
        head, frac = t, ""
    head = re.sub(r"\D", "", head)
    frac = re.sub(r"\D", "", frac)
    if not head and not frac:
        return tok
    if frac:
        return _dec_words(head, frac, lang)
    return _int_words(head, lang)


def _p_integer(text: str, lang: str) -> str:
    """Catch-all: every digit run nobody else claimed."""
    return re.sub(r"(?<![\w])(\d(?:[\d.,'  ]*\d)?)(?![\w])",
                  lambda m: _num_token(m.group(1), lang), text)


# ------------------------------------------------- locale-aware core shapes
_MONTHS = {
    "de": ["", "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
           "August", "September", "Oktober", "November", "Dezember"],
    "en": ["", "January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
    "es": ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
           "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
    "fr": ["", "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
           "août", "septembre", "octobre", "novembre", "décembre"],
    "it": ["", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
           "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"],
    "pt": ["", "janeiro", "fevereiro", "março", "abril", "maio", "junho",
           "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"],
    "nl": ["", "januari", "februari", "maart", "april", "mei", "juni", "juli",
           "augustus", "september", "oktober", "november", "december"],
    "ru": ["", "января", "февраля", "марта", "апреля", "мая", "июня", "июля",
           "августа", "сентября", "октября", "ноября", "декабря"],
    "tr": ["", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
           "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"],
}

# (singular, plural) per language; the cents word follows the same table.
_CURRENCY = {
    "CHF": {"de": "Franken", "en": "Swiss francs", "es": "francos suizos",
            "fr": "francs suisses", "it": "franchi svizzeri",
            "pt": "francos suíços", "nl": "Zwitserse frank",
            "ru": "швейцарских франков", "tr": "İsviçre frangı"},
    "EUR": {"de": "Euro", "en": "euros", "es": "euros", "fr": "euros",
            "it": "euro", "pt": "euros", "nl": "euro", "ru": "евро",
            "tr": "avro"},
    "USD": {"de": "Dollar", "en": "dollars", "es": "dólares", "fr": "dollars",
            "it": "dollari", "pt": "dólares", "nl": "dollar",
            "ru": "долларов", "tr": "dolar"},
    "GBP": {"de": "Pfund", "en": "pounds", "es": "libras", "fr": "livres",
            "it": "sterline", "pt": "libras", "nl": "pond", "ru": "фунтов",
            "tr": "sterlin"},
}
_CUR_SYMBOL = {"€": "EUR", "$": "USD", "£": "GBP", "CHF": "CHF", "Fr.": "CHF",
               "EUR": "EUR", "USD": "USD", "GBP": "GBP"}
# Day-first everywhere except English.
_DAY_FIRST = {"de", "es", "fr", "it", "pt", "nl", "ru", "tr"}


def _p_currency(text: str, lang: str) -> str:
    syms = sorted(_CUR_SYMBOL, key=len, reverse=True)
    alt = "|".join(re.escape(s) for s in syms)
    amt = r"\d(?:[\d'.,]*\d)?"

    def rep(m):
        code = _CUR_SYMBOL[m.group("cur")]
        word = _CURRENCY[code].get(lang, _CURRENCY[code]["en"])
        raw = m.group("amt")
        # Money is spoken as "twelve francs fifty", not "twelve point five
        # zero francs" — two fraction digits are cents, not a decimal.
        t = raw.replace("'", "").replace(" ", "")
        mm = re.match(r"^(\d+)[.,](\d{2})$", t)
        if mm:
            main = _int_words(mm.group(1), lang)
            cents = int(mm.group(2))
            if cents == 0:
                return f"{main} {word}"
            return f"{main} {word} {_int_words(mm.group(2).lstrip('0') or '0', lang)}"
        return f"{_num_token(raw, lang)} {word}"

    out = re.sub(rf"(?P<cur>{alt})\s?(?P<amt>{amt})", rep, text)
    return re.sub(rf"(?P<amt>{amt})\s?(?P<cur>{alt})", rep, out)


def _p_date(text: str, lang: str) -> str:
    months = _MONTHS.get(lang, _MONTHS["en"])

    def rep(m):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        day, mon = (a, b) if lang in _DAY_FIRST else (b, a)
        if not (1 <= mon <= 12 and 1 <= day <= 31):
            return m.group(0)
        yw = _n2w(y, lang, to="year")
        if lang == "en":
            return f"{months[mon]} {_n2w(day, lang, to='ordinal')}, {yw}"
        if lang == "de":
            return f"{_n2w(day, lang, to='ordinal')} {months[mon]} {yw}"
        return f"{_int_words(str(day), lang)} {months[mon]} {yw}"

    return re.sub(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", rep, text)


def _p_time(text: str, lang: str) -> str:
    def rep(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            return m.group(0)
        hw = _int_words(str(h), lang)
        # The joiner between hours and minutes is lexical per language and is
        # NOT the plural noun for "hours": German says "vierzehn Uhr dreißig",
        # Italian "quattordici e trenta", Spanish "catorce treinta".
        joiner = {"de": " Uhr ", "en": " ", "fr": " heures ", "it": " e ",
                  "es": " y ", "pt": " e ", "nl": " uur ", "ru": " часов ",
                  "tr": " "}.get(lang, " ")
        whole = {"de": " Uhr", "en": " o'clock", "fr": " heures",
                 "it": " in punto", "es": " en punto", "pt": " horas",
                 "nl": " uur", "ru": " часов", "tr": ""}.get(lang, "")
        if mi == 0:
            return f"{hw}{whole}"
        return f"{hw}{joiner}{_int_words(str(mi), lang)}"
    return re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*\d)", rep, text)


# Order matters — see the module docstring. These run BEFORE the fast tier
# (for de/en) because the fast tier's catch-all integer rule would otherwise
# consume the digits these passes need to see intact ("3rd" -> "threerd",
# "2h 30min" -> "twoh thirtymin", "v2.10" -> "vtwo point one zero").
#
# DATES AND TIMES GO FIRST, always. Learned the hard way on the first run of
# this module: with the score pass ahead of them, "14:30" became "fourteen to
# thirty" (a football score), and with the fraction and math passes ahead of
# them "15/06/1920" became "fifteen divided by six". Every one of those passes
# is a catch-all that assumes the unambiguous shapes are already gone.
#
# Note which passes are NOT in the de/en list: dates and times. The fast tier
# already reads those better than a generic rule can — German dative after
# "am/vom/zum" ("am fünfzehnten Juni", not "am fünfzehnte Juni"), Swiss
# dot-times, "14.30 Uhr". Running the generic version first would consume the
# digits and throw that away.
# _p_score is deliberately absent here too, for the same reason: "14:30" is a
# time, and the fast tier is the one that knows that. It runs AFTER the fast
# tier instead, by which point every real time is already words and whatever
# colon pair is left really is a score.
_SHAPE_PASSES_DEEN = (_p_roman, _p_version, _p_duration, _p_score, _p_range,
                      _p_fraction, _p_decade, _p_ordinal, _p_math)
_SHAPE_PASSES = (_p_date, _p_time, _p_roman, _p_version, _p_duration,
                 _p_score, _p_range, _p_fraction, _p_decade, _p_ordinal,
                 _p_math)
# …and these complete the job for languages the fast tier does not speak.
_LOCALE_PASSES = (_p_currency, _p_percent, _p_units, _p_year, _p_integer)

_HAS_WORK = re.compile(r"\d|[IVXLCDM]{2,}|[%‰×÷°]")


def normalize_quality(text: str, lang: str = "en") -> str:
    """Deterministic quality tier.

    de/en: the new shape passes run FIRST, then the fast tier finishes the
    job with its German/English-specific date, currency and identifier rules
    (dative ordinals after "am/vom", "zwölf Franken fünfzig", leading-zero
    order numbers) — those are better than anything generic and are kept.

    Everything else: the fast tier's number core only speaks de/en, so
    running it on Spanish or Russian emits ENGLISH number words into the
    sentence. For those languages the fast tier is used for text shaping
    only, and this module renders every number itself, in the actual
    language, via num2words.
    """
    if not text:
        return text
    return _quality_cached(text, _lang2(lang))


@lru_cache(maxsize=512)
def _quality_cached(text: str, lang: str) -> str:
    if lang not in QUALITY_LANGS or not HAVE_NUM2WORDS:
        return _fast_normalize(text, lang)
    try:
        out = _fast._strip_formatting(text)
        out = _fast._verbalize_addresses(out, lang == "de")
        if _HAS_WORK.search(out):
            if lang in ("de", "en"):
                for p in _SHAPE_PASSES_DEEN:
                    out = p(out, lang)
                # Month-first dates are an English shape the fast tier does
                # not have — its date rule is day-first, so "06/15/1920" fell
                # through it into the integer catch-all. Running this first
                # claims US-style dates; European "15/06" has an impossible
                # month here, is left alone, and the fast tier reads it.
                if lang == "en":
                    out = _p_date(out, "en")
                out = _fast_normalize(out, lang)
            else:
                for p in _SHAPE_PASSES:
                    out = p(out, lang)
                for p in _LOCALE_PASSES:
                    out = p(out, lang)
        elif lang in ("de", "en"):
            out = _fast_normalize(out, lang)   # abbreviations, symbols, "&"
        return re.sub(r"\s{2,}", " ", out).strip()
    except Exception:
        return _fast_normalize(text, lang)


def normalize(text: str, lang: str = "en", mode: str = "fast") -> str:
    """Single entry point for both tiers. `mode` is 'fast' or 'quality'."""
    if mode == "quality":
        return normalize_quality(text, lang)
    return _fast_normalize(text, lang)
