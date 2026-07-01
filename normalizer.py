"""
Text normalizer for TTS - DE + EN. Regex-based, ALWAYS-ON safety net.

NeMo / pynini are intentionally NOT used (banned: Windows pynini issues).
mT5 path is omitted: google/mt5-300m is not fine-tuned for TTS normalization
and there is no confirmed DE/TTS normalization checkpoint on HF, so loading a
300M model would add latency/RAM for no quality gain. The regex normalizer is
robust and never raises - on any error it returns the original text unchanged.

Handles: currency (CHF/EUR/USD/GBP), dates (15.06.2026), times (14:30 Uhr),
common abbreviations (Dr., Nr., ca., z.B., etc.), units (km, kg, GB, ...),
and number-to-words for integers and decimals.
"""
from __future__ import annotations
import re

# ---------------------------------------------------------------- number words
_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"]
_EN_SCALE = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]

_DE_ONES = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
            "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
            "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
_DE_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig",
            "siebzig", "achtzig", "neunzig"]


def _en_below_1000(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        t = _EN_TENS[n // 10]
        return t if n % 10 == 0 else f"{t}-{_EN_ONES[n % 10]}"
    h = f"{_EN_ONES[n // 100]} hundred"
    return h if n % 100 == 0 else f"{h} {_en_below_1000(n % 100)}"


def en_number(n: int) -> str:
    if n == 0:
        return "zero"
    neg = n < 0
    n = abs(n)
    parts: list[str] = []
    for value, name in _EN_SCALE:
        if n >= value:
            parts.append(f"{_en_below_1000(n // value)} {name}")
            n %= value
    if n:
        parts.append(_en_below_1000(n))
    out = " ".join(parts)
    return ("minus " + out) if neg else out


def _de_below_100(n: int) -> str:
    if n < 20:
        return _DE_ONES[n]
    tens = _DE_TENS[n // 10]
    o = n % 10
    if o == 0:
        return tens
    ones = "ein" if o == 1 else _DE_ONES[o]
    return f"{ones}und{tens}"


def _de_below_1000(n: int) -> str:
    if n < 100:
        return _de_below_100(n)
    h = ("ein" if n // 100 == 1 else _DE_ONES[n // 100]) + "hundert"
    return h if n % 100 == 0 else h + _de_below_100(n % 100)


def de_number(n: int) -> str:
    if n == 0:
        return "null"
    neg = n < 0
    n = abs(n)
    out = ""
    if n >= 1_000_000:
        millions = n // 1_000_000
        out += ("eine Million " if millions == 1 else f"{_de_below_1000(millions)} Millionen ")
        n %= 1_000_000
    if n >= 1000:
        th = n // 1000
        out += ("eintausend" if th == 1 else f"{_de_below_1000(th)}tausend")
        n %= 1000
    if n:
        out += _de_below_1000(n)
    out = out.strip()
    return ("minus " + out) if neg else out


def number_to_words(n: int, lang: str) -> str:
    return de_number(n) if lang.startswith("de") else en_number(n)


def _decimal_to_words(int_part: int, frac: str, lang: str) -> str:
    sep = "Komma" if lang.startswith("de") else "point"
    digits = " ".join(number_to_words(int(d), lang) for d in frac)
    return f"{number_to_words(int_part, lang)} {sep} {digits}"


# ---------------------------------------------------------------- lexicons
_ABBREV = {
    "de": {
        "z.B.": "zum Beispiel", "z. B.": "zum Beispiel", "u.a.": "unter anderem",
        "usw.": "und so weiter", "bzw.": "beziehungsweise", "ca.": "circa",
        "Dr.": "Doktor", "Prof.": "Professor", "Nr.": "Nummer", "Abb.": "Abbildung",
        "Tel.": "Telefon", "Str.": "Straße", "MwSt.": "Mehrwertsteuer",
        "inkl.": "inklusive", "exkl.": "exklusive", "evtl.": "eventuell",
        "max.": "maximal", "min.": "minimal", "vgl.": "vergleiche",
    },
    "en": {
        "e.g.": "for example", "i.e.": "that is", "etc.": "et cetera",
        "Dr.": "Doctor", "Prof.": "Professor", "No.": "number", "Mr.": "Mister",
        "Mrs.": "Misses", "Ms.": "Miss", "vs.": "versus", "approx.": "approximately",
        "Fig.": "figure", "Inc.": "Incorporated", "Ltd.": "Limited",
    },
}
_UNITS = {
    "de": {"km": "Kilometer", "cm": "Zentimeter", "mm": "Millimeter", "kg": "Kilogramm",
           "GB": "Gigabyte", "MB": "Megabyte", "TB": "Terabyte", "kWh": "Kilowattstunden",
           "km/h": "Kilometer pro Stunde", "°C": "Grad Celsius"},
    "en": {"km": "kilometers", "cm": "centimeters", "mm": "millimeters", "kg": "kilograms",
           "GB": "gigabytes", "MB": "megabytes", "TB": "terabytes", "kWh": "kilowatt hours",
           "km/h": "kilometers per hour", "°C": "degrees Celsius"},
}
_CURRENCY = {
    "CHF": ("Franken", "francs"), "€": ("Euro", "euros"), "EUR": ("Euro", "euros"),
    "$": ("Dollar", "dollars"), "USD": ("Dollar", "dollars"),
    "£": ("Pfund", "pounds"), "GBP": ("Pfund", "pounds"),
}
_MONTHS_DE = ["", "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
              "August", "September", "Oktober", "November", "Dezember"]
_MONTHS_EN = ["", "January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]


def normalize(text: str, lang: str = "en", enabled: bool = True) -> str:
    if not enabled or not text:
        return text
    try:
        de = lang.startswith("de")
        out = text

        # abbreviations (longest first so "z. B." beats "z.")
        for abbr, full in sorted(_ABBREV["de" if de else "en"].items(), key=lambda x: -len(x[0])):
            out = out.replace(abbr, full)

        # currency: symbol/code before or after the amount
        def _money(m):
            cur = m.group("cur")
            amount = m.group("amt")
            names = _CURRENCY.get(cur)
            word = (names[0] if de else names[1]) if names else cur
            spoken = _spell_amount(amount, lang)
            return f"{spoken} {word}"

        cur_alt = "|".join(re.escape(c) for c in _CURRENCY)
        out = re.sub(rf"(?P<cur>{cur_alt})\s?(?P<amt>\d[\d'.,]*)", _money, out)
        out = re.sub(rf"(?P<amt>\d[\d'.,]*)\s?(?P<cur>{cur_alt})", _money, out)

        # dates dd.mm.yyyy or dd/mm/yyyy
        def _date(m):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if not (1 <= mo <= 12 and 1 <= d <= 31):
                return m.group(0)
            month = _MONTHS_DE[mo] if de else _MONTHS_EN[mo]
            if de:
                return f"{number_to_words(d, lang)}. {month} {number_to_words(y, lang)}"
            return f"{month} {en_number(d)}, {en_number(y)}"
        out = re.sub(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", _date, out)

        # times HH:MM (optional 'Uhr')
        def _time(m):
            h, mi = int(m.group(1)), int(m.group(2))
            if h > 23 or mi > 59:
                return m.group(0)
            if de:
                base = f"{number_to_words(h, lang)} Uhr"
                return base if mi == 0 else f"{base} {number_to_words(mi, lang)}"
            if mi == 0:
                return f"{en_number(h)} o'clock"
            mm = f"oh {en_number(mi)}" if mi < 10 else en_number(mi)
            return f"{en_number(h)} {mm}"
        # consume an optional trailing 'Uhr' so we don't say "Uhr" twice
        out = re.sub(r"\b(\d{1,2}):(\d{2})\b(?:\s*Uhr)?", _time, out)

        # units: number + unit
        unit_map = _UNITS["de" if de else "en"]
        unit_alt = "|".join(re.escape(u) for u in sorted(unit_map, key=lambda x: -len(x)))
        def _unit(m):
            spoken = _spell_amount(m.group(1), lang)
            return f"{spoken} {unit_map[m.group(2)]}"
        out = re.sub(rf"(\d[\d'.,]*)\s?({unit_alt})\b", _unit, out)

        # any remaining standalone numbers / decimals
        def _num(m):
            return _spell_amount(m.group(0), lang)
        out = re.sub(r"\d[\d'.,]*\d|\d", _num, out)

        return re.sub(r"\s{2,}", " ", out).strip()
    except Exception:  # noqa: BLE001 - never break synthesis
        return text


def _spell_amount(token: str, lang: str) -> str:
    """
    Turn a raw numeric token into words, disambiguating thousands/decimal
    separators. Apostrophes are always thousands (Swiss: 1'234.50). When both
    '.' and ',' appear, the last one is the decimal point. A lone '.' in German
    is treated as thousands only when it groups exactly 3 digits, so "3.5" stays
    a decimal but "1.234" is a thousand.
    """
    t = token.strip().replace("'", "").replace(" ", "").replace(" ", "")
    de = lang.startswith("de")
    int_str, frac = t, ""
    has_dot, has_com = "." in t, "," in t
    if has_dot and has_com:
        dec = "." if t.rfind(".") > t.rfind(",") else ","
        int_str, _, frac = t.rpartition(dec)
        int_str = int_str.replace(".", "").replace(",", "")
    elif has_com:
        if de:
            int_str, _, frac = t.rpartition(",")
        else:
            int_str = t.replace(",", "")
    elif has_dot:
        head, _, tail = t.rpartition(".")
        if not de:
            int_str, frac = head, tail
        elif len(tail) == 3 and head.replace(".", "").isdigit():
            int_str = t.replace(".", "")
        else:
            int_str, frac = head, tail
    int_str = re.sub(r"\D", "", int_str)
    frac = re.sub(r"\D", "", frac)
    if not int_str:
        return token
    if frac:
        return _decimal_to_words(int(int_str), frac, lang)
    return number_to_words(int(int_str), lang)
