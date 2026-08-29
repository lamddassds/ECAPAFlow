"""
Text normalizer for TTS - DE + EN. Regex-based, ALWAYS-ON safety net.

NeMo / pynini are intentionally NOT used (banned: Windows pynini issues).
mT5 path is omitted: google/mt5-300m is not fine-tuned for TTS normalization
and there is no confirmed DE/TTS normalization checkpoint on HF, so loading a
300M model would add latency/RAM for no quality gain. The regex normalizer is
robust and never raises - on any error it returns the original text unchanged.

Handles: currency (CHF/EUR/USD/GBP), dates (15.06.2026), times (14:30 Uhr),
common abbreviations (Dr., Nr., ca., z.B., etc.), units (km, kg, GB, ...),
number-to-words for integers and decimals, leading-zero identifiers
("016" -> digit-by-digit), and stripping Markdown/emoji/junk punctuation
that would otherwise be read aloud as literal garbage ("asterisk asterisk").
"""
from __future__ import annotations
import re
from functools import lru_cache

# ---------------------------------------------------------------- formatting
# Confirmed real, repeatedly-reported failure across TTS products (raw
# **bold**/#headers/etc. get read as literal symbols, e.g. "asterisk
# asterisk") — chatbot/article/announcement content plausibly comes from
# LLM output or pasted docs, so this isn't an edge case. Runs FIRST, before
# abbreviations/numbers, so stripped junk can't create bogus matches and
# _SENT_SPLIT_RE (applied after normalize()) sees clean sentence punctuation.
_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")
_SMART_QUOTES_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
})
_MD_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|___)(.+?)\1")
_MD_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_MD_ITALIC_RE = re.compile(r"(?<!\w)([*_])(?!\1)(.+?)\1(?!\w)")
_MD_HEADER_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_MD_QUOTE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE)
_MD_HR_RE = re.compile(r"^[ \t]{0,3}([-*_])\1{2,}[ \t]*$", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^[ \t]{0,3}[-*+][ \t]+", re.MULTILINE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    "]+"
)
_REPEAT_BANG_RE = re.compile(r"([!?])\1+")
_RUNAWAY_DOTS_RE = re.compile(r"\.{4,}")
_SPACED_ELLIPSIS_RE = re.compile(r"\.\s\.\s\.")


_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*"
    r"@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")
# ends on a non-terminator char so a trailing sentence period stays a
# sentence period instead of being read as part of the address
_URL_RE = re.compile(r"\bhttps?://\S*[^\s.,;:!?]|\bwww\.\S*[^\s.,;:!?]")


def _verbalize_addresses(text: str, de: bool) -> str:
    """Emails/URLs read aloud, applied PER TOKEN with a length guard.

    A full-text regex scan is quadratic on pathological dotted runs
    ("user@" + "a."*N reached 15-45s at the 100k text cap; adversarial
    review 2026-07-05, both the class-based and label-chained patterns).
    Tokenizing first bounds every regex call to one whitespace-delimited
    token, and the length guard (RFC 5321 caps real addresses well below
    it) skips degenerate mega-tokens entirely — worst case is linear.
    """
    if "@" not in text and "www." not in text and "http" not in text:
        return text
    dot_w = " Punkt " if de else " dot "
    at_w = " ät " if de else " at "
    slash_w = " Slash " if de else " slash "

    def _email(m):
        return m.group(0).replace("@", at_w).replace(".", dot_w)

    def _url(m):
        u = re.sub(r"^https?://", "", m.group(0))
        return u.replace("/", slash_w).replace(".", dot_w)

    parts = text.split(" ")
    for i, tok in enumerate(parts):
        if not tok or len(tok) > 2048:
            continue
        if "@" in tok:
            parts[i] = _EMAIL_RE.sub(_email, tok)
        elif "www." in tok or "http" in tok:
            parts[i] = _URL_RE.sub(_url, tok)
    return " ".join(parts)


def _strip_formatting(text: str) -> str:
    """Markdown/emoji/junk-punctuation cleanup — see module docstring.
    Order matters: code/links/images first, so their inner */_ don't get
    caught by the bold/italic pass. Bullet markers ("- "/"* "/"+ ") are
    stripped; NUMBERED list markers ("1. ") are deliberately NOT stripped —
    they collide with real German ordinal constructions ("1. Mal", "3.
    Platz") that must not lose their meaning."""
    out = _ZERO_WIDTH_RE.sub("", text)
    out = out.translate(_SMART_QUOTES_MAP)
    out = _MD_CODE_BLOCK_RE.sub(" ", out)
    out = _MD_INLINE_CODE_RE.sub(r"\1", out)
    out = _MD_IMAGE_RE.sub(" ", out)
    out = _MD_LINK_RE.sub(r"\1", out)
    out = _MD_BOLD_ITALIC_RE.sub(r"\2", out)
    out = _MD_BOLD_RE.sub(r"\2", out)
    out = _MD_ITALIC_RE.sub(r"\2", out)
    out = _MD_HEADER_RE.sub("", out)
    out = _MD_QUOTE_RE.sub("", out)
    out = _MD_HR_RE.sub(" ", out)
    out = _MD_BULLET_RE.sub("", out)
    out = _EMOJI_RE.sub(" ", out)
    out = _SPACED_ELLIPSIS_RE.sub("...", out)
    out = _RUNAWAY_DOTS_RE.sub("...", out)
    out = _REPEAT_BANG_RE.sub(r"\1", out)
    return out

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


def digits_spelled(digits: str, lang: str) -> str:
    """Read a digit string one digit at a time: "016" -> "null eins sechs".
    For identifiers (order/reference numbers, room numbers, phone-number
    groups), not quantities — see the leading-zero rule in _spell_amount."""
    ones = _DE_ONES if lang.startswith("de") else _EN_ONES
    return " ".join(ones[int(d)] for d in digits if d.isdigit())


def en_year(n: int) -> str:
    """Years read as years: 1920 -> "nineteen twenty", 1905 -> "nineteen oh
    five", 1900 -> "nineteen hundred", 2015 -> "twenty fifteen". 2000-2009
    and anything outside the year ranges fall back to the cardinal."""
    hi, lo = divmod(n, 100)
    if 1100 <= n <= 1999:
        if lo == 0:
            return f"{_en_below_1000(hi)} hundred"
        if lo < 10:
            return f"{_en_below_1000(hi)} oh {_EN_ONES[lo]}"
        return f"{_en_below_1000(hi)} {_en_below_1000(lo)}"
    if 2010 <= n <= 2099:
        return f"{_en_below_1000(hi)} {_en_below_1000(lo)}"
    return en_number(n)


def de_year(n: int) -> str:
    """1920 -> "neunzehnhundertzwanzig", 1905 -> "neunzehnhundertfünf".
    From 2000 on the German cardinal is already the correct year form."""
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        base = _de_below_100(hi) + "hundert"
        return base if lo == 0 else base + _de_below_100(lo)
    return de_number(n)


def year_to_words(n: int, lang: str) -> str:
    return de_year(n) if lang.startswith("de") else en_year(n)


_EN_ORD_SPECIAL = {1: "first", 2: "second", 3: "third", 5: "fifth",
                   8: "eighth", 9: "ninth", 12: "twelfth"}


def en_ordinal(n: int) -> str:
    """Ordinal words for dates: 1 -> "first", 21 -> "twenty-first"."""
    if n in _EN_ORD_SPECIAL:
        return _EN_ORD_SPECIAL[n]
    if n < 20:
        return _EN_ONES[n] + "th"
    if n % 10 == 0:
        return _EN_TENS[n // 10][:-1] + "ieth"
    return f"{_EN_TENS[n // 10]}-{en_ordinal(n % 10)}"


_DE_ORD_STEM = {1: "erst", 3: "dritt", 7: "siebt", 8: "acht"}


def de_ordinal(n: int, suffix: str = "er") -> str:
    """German date ordinals: de_ordinal(15, "en") -> "fünfzehnten" (after
    "am"/"vom"), de_ordinal(15) -> "fünfzehnter" (nominative)."""
    stem = _DE_ORD_STEM.get(n)
    if stem is None:
        stem = (_DE_ONES[n] + "t") if n < 20 else (_de_below_100(n) + "st")
    return stem + suffix


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
        "d.h.": "das heißt", "d. h.": "das heißt", "ggf.": "gegebenenfalls",
        "bzgl.": "bezüglich", "sog.": "sogenannte", "bspw.": "beispielsweise",
        "Mio.": "Millionen", "Mrd.": "Milliarden", "Jh.": "Jahrhundert",
        "u.v.m.": "und vieles mehr", "St.": "Sankt", "Hr.": "Herr",
        # "Fr." is deliberately NOT here: in Swiss texts it means francs
        # ("Fr. 20.50") and is handled by the currency rule instead.
    },
    "en": {
        "e.g.": "for example", "i.e.": "that is", "etc.": "et cetera",
        "Dr.": "Doctor", "Prof.": "Professor", "No.": "number", "Mr.": "Mister",
        "Mrs.": "Misses", "Ms.": "Miss", "vs.": "versus", "approx.": "approximately",
        "Fig.": "figure", "Inc.": "Incorporated", "Ltd.": "Limited",
        "St.": "Saint", "Jr.": "Junior", "Sr.": "Senior", "Ave.": "Avenue",
        "Dept.": "Department", "Corp.": "Corporation", "Mt.": "Mount",
    },
}
_UNITS = {
    "de": {"km": "Kilometer", "cm": "Zentimeter", "mm": "Millimeter", "kg": "Kilogramm",
           "GB": "Gigabyte", "MB": "Megabyte", "TB": "Terabyte", "kWh": "Kilowattstunden",
           "km/h": "Kilometer pro Stunde", "°C": "Grad Celsius", "m": "Meter",
           "g": "Gramm", "mg": "Milligramm", "ml": "Milliliter",
           "km²": "Quadratkilometer", "m²": "Quadratmeter", "ms": "Millisekunden",
           "Hz": "Hertz", "kHz": "Kilohertz", "MHz": "Megahertz", "GHz": "Gigahertz"},
    "en": {"km": "kilometers", "cm": "centimeters", "mm": "millimeters", "kg": "kilograms",
           "GB": "gigabytes", "MB": "megabytes", "TB": "terabytes", "kWh": "kilowatt hours",
           "km/h": "kilometers per hour", "°C": "degrees Celsius", "m": "meters",
           "g": "grams", "mg": "milligrams", "ml": "milliliters",
           "km²": "square kilometers", "m²": "square meters", "ms": "milliseconds",
           "Hz": "hertz", "kHz": "kilohertz", "MHz": "megahertz", "GHz": "gigahertz"},
}
_CURRENCY = {
    "CHF": ("Franken", "francs"), "Fr.": ("Franken", "francs"),
    "€": ("Euro", "euros"), "EUR": ("Euro", "euros"),
    "$": ("Dollar", "dollars"), "USD": ("Dollar", "dollars"),
    "£": ("Pfund", "pounds"), "GBP": ("Pfund", "pounds"),
}
_MONTHS_DE = ["", "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
              "August", "September", "Oktober", "November", "Dezember"]
_MONTHS_EN = ["", "January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------- fast path
# Two cheap wins ported from QuantFlow's normalizer (2026-07-27), both purely
# performance — the output is unchanged in every case:
#
# 1. EARLY-EXIT PROBES. Ordinary prose contains nothing this module rewrites,
#    yet it used to pay for every regex pass anyway. The probes below decide
#    in one scan each whether there is anything to do. QuantFlow's recorded
#    bug is the trap to avoid: its first abbreviation probe included a bare
#    "\." and therefore matched every sentence-ending period, so the probe
#    never skipped anything. These enumerate actual shapes instead.
# 2. MEMOIZATION. normalize() is a pure function of (text, lang), and this app
#    re-normalizes the same sentences constantly (retakes, A/B runs, the
#    Regenerate button, per-voice previews). The whitespace collapse lives
#    INSIDE the cached function so the cached value is the finished string.
_HAS_DIGIT = re.compile(r"\d")
_SYMBOL_PROBE = re.compile(r"[&%@°€$£+×/]|https?://|www\.|\b[IVXLC]{2,}\b")
_ABBREV_PROBE_CACHE: dict[str, "re.Pattern[str]"] = {}


def _abbrev_probe(de: bool) -> "re.Pattern[str]":
    """One alternation over the real abbreviation keys, compiled once."""
    key = "de" if de else "en"
    pat = _ABBREV_PROBE_CACHE.get(key)
    if pat is None:
        keys = sorted(_ABBREV[key], key=len, reverse=True)
        pat = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")")
        _ABBREV_PROBE_CACHE[key] = pat
    return pat


def normalize(text: str, lang: str = "en", enabled: bool = True) -> str:
    if not enabled or not text:
        return text
    return _normalize_cached(text, lang)


# The number core of this module speaks German and English and nothing else.
# Applied to any other language it does not decline to act — it writes ENGLISH
# number words into the sentence ("Это 12 км" -> "Это twelve kilometers"),
# which is worse than leaving the digits: Supertonic reads bare digits in the
# target language perfectly well. So outside de/en this tier does text shaping
# only, and the multilingual rendering lives in normalizer_quality.py, which
# has num2words behind it. Found 2026-07-27 while building that tier.
NUMERIC_LANGS = {"de", "en"}


@lru_cache(maxsize=512)
def _normalize_cached(text: str, lang: str = "en") -> str:
    if lang[:2].lower() not in NUMERIC_LANGS:
        try:
            return re.sub(r"\s{2,}", " ", _strip_formatting(text)).strip()
        except Exception:  # noqa: BLE001 - never break synthesis
            return text
    try:
        # Probe the STRIPPED text, never the raw text. Zero-width characters,
        # smart quotes and markdown all sit between an abbreviation and its
        # pattern: "z.​B." does not match \bz\.B\. until the zero-width
        # space is gone, so probing raw input silently skipped a real
        # expansion (caught by tests/test_normalizer.py the same day this
        # fast path was added).
        stripped = _strip_formatting(text)
        if (not _HAS_DIGIT.search(stripped)
                and not _SYMBOL_PROBE.search(stripped)
                and not _abbrev_probe(lang.startswith("de")).search(stripped)):
            # Nothing normalizable — but the formatting strip and whitespace
            # collapse still have to happen, since callers rely on them.
            return re.sub(r"\s{2,}", " ", stripped).strip()
    except Exception:  # noqa: BLE001 - a probe must never break synthesis
        pass
    try:
        de = lang.startswith("de")
        out = _strip_formatting(text)

        # Emails and explicit URLs, read the way people actually say them
        # ("info@firma.de" -> "info ät firma Punkt de"; ElevenLabs' own
        # normalizer does the same "john at gmail dot com" verbalization).
        # Runs FIRST (before abbreviations/numbers) so nothing else mangles
        # the token. Deliberately conservative: only unambiguous shapes
        # (something@domain.tld, http(s)://…, www.…) — bare "example.com"
        # style domains are left for the model, to avoid false positives.
        out = _verbalize_addresses(out, de)

        # abbreviations (longest first so "z. B." beats "z."). A leading \b
        # is required — bare str.replace matched "min." inside "Termin."
        # ("Der Termin." -> "Der Terminimal") and "ca." inside "Africa."
        # ("Africa." -> "Africirca"); confirmed audible bug, 2026-07-03.
        if _abbrev_probe(de).search(out):
            for abbr, full in sorted(_ABBREV["de" if de else "en"].items(),
                                     key=lambda x: -len(x[0])):
                out = re.sub(r"\b" + re.escape(abbr), full, out)

        # currency: symbol/code before or after the amount. Two-digit
        # fractions read like money is spoken: "CHF 12.50" -> "zwölf
        # Franken fünfzig", not "zwölf Komma fünf null Franken".
        def _money(m):
            cur = m.group("cur")
            amount = m.group("amt")
            names = _CURRENCY.get(cur)
            word = (names[0] if de else names[1]) if names else cur
            int_str, frac = _parse_number(amount, lang)
            if int_str and len(frac) == 2:
                main = number_to_words(int(int_str), lang)
                if int(frac) == 0:
                    return f"{main} {word}"
                return f"{main} {word} {number_to_words(int(frac), lang)}"
            return f"{_spell_amount(amount, lang)} {word}"

        # amounts must END on a digit, or a trailing sentence period gets
        # swallowed ("CHF 1'234.50." would otherwise read as 123450)
        cur_alt = "|".join(re.escape(c) for c in _CURRENCY)
        amt = r"\d(?:[\d'.,]*\d)?"
        out = re.sub(rf"(?P<cur>{cur_alt})\s?(?P<amt>{amt})", _money, out)
        out = re.sub(rf"(?P<amt>{amt})\s?(?P<cur>{cur_alt})", _money, out)

        # dates dd.mm.yyyy or dd/mm/yyyy — days become ordinals. In German
        # a preceding "am/vom/zum/bis" selects the dative form ("am
        # fünfzehnten Juni"), otherwise nominative ("fünfzehnter Juni").
        def _date(m):
            prep = m.group(1)
            d, mo, y = int(m.group(2)), int(m.group(3)), int(m.group(4))
            if not (1 <= mo <= 12 and 1 <= d <= 31):
                return m.group(0)
            month = _MONTHS_DE[mo] if de else _MONTHS_EN[mo]
            head = f"{prep} " if prep else ""
            if de:
                day = de_ordinal(d, "en" if prep else "er")
                return f"{head}{day} {month} {year_to_words(y, lang)}"
            return f"{head}{month} {en_ordinal(d)}, {en_year(y)}"
        out = re.sub(
            r"\b(?:(am|vom|zum|bis)\s+)?(\d{1,2})[./](\d{1,2})[./](\d{4})\b",
            _date, out, flags=re.IGNORECASE)

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

        # German dot-times ("um 14.30 Uhr", common in CH/DE) would otherwise
        # be read as the decimal "vierzehn Komma dreißig Uhr"
        if de:
            def _dottime(m):
                h, mi = int(m.group(1)), int(m.group(2))
                if h > 23 or mi > 59:
                    return m.group(0)
                base = f"{number_to_words(h, lang)} Uhr"
                return base if mi == 0 else f"{base} {number_to_words(mi, lang)}"
            out = re.sub(r"\b(\d{1,2})\.(\d{2})\s*Uhr\b", _dottime, out)

        # German ordinals in running prose: "das 1. Mal" -> "das erste Mal",
        # "im 3. Stock" -> "im dritten Stock", "der 2. Weltkrieg" -> "der
        # zweite Weltkrieg", "am 15. Juni" -> "am fünfzehnten Juni" (numeric
        # dd.mm.yyyy dates were already consumed by the date rule above).
        # CONSERVATIVE by design: only fires when a preposition/article that
        # pins the grammatical case precedes "N." AND a capitalized noun
        # follows — German adjective declension after a definite article is
        # "-e" in nominative/accusative and "-en" in dative/genitive, so
        # those two suffixes cover the clear cases. "der" alone is ambiguous
        # (nominative masculine -e vs dative/genitive feminine -en); a known
        # dative/genitive preposition before it selects -en. Anything the
        # rule can't pin stays a cardinal (the previous behavior) rather
        # than risking a wrong ending, which would sound worse.
        if de:
            def _ord_prose(m):
                prep, before, art = m.group(1), m.group(2), m.group(3)
                n, noun = int(m.group(4)), m.group(5)
                if not (1 <= n <= 99):
                    return m.group(0)
                low = art.lower()
                if low == "der":
                    # bare "der" is nominative masculine ("-e") UNLESS an
                    # oblique case is pinned: a dative/genitive preposition
                    # before it, or a directly preceding noun — the genitive
                    # chain "Gewinner der 3. Runde" / "Ende der 1. Halbzeit"
                    # (adversarial review 2026-07-05). "der" lowercase only:
                    # sentence-initial "Der" is safely nominative.
                    suffix = "en" if (prep or (before and art == "der")) else "e"
                elif low in ("die", "das"):
                    # "-en"-plural nouns after "die" need the plural ending
                    # ("die 3. Klassen" -> "die dritten Klassen"); other
                    # plural shapes (-e/-er/umlaut) are indistinguishable
                    # from singulars by surface form — accepted residual.
                    suffix = "en" if (low == "die" and noun.endswith("en")) else "e"
                else:  # am/im/vom/zum/zur/beim/dem/den/des — dative/genitive
                    suffix = "en"
                head = f"{before} " if before else (f"{prep} " if prep else "")
                return f"{head}{art} {de_ordinal(n, suffix)} {noun}"
            # The lookahead blocks capitalized FUNCTION words after "N." —
            # those start a new sentence ("Ich tippe auf die 3. Das ist…"),
            # where N is a cardinal and rewriting would merge two sentences
            # and change meaning (worst failure class of the adversarial
            # review). An attributive ordinal is always followed by a noun.
            out = re.sub(
                r"(?:\b((?i:seit|von|mit|nach|aus|bei|in|an|auf|vor|ab"
                r"|gegenüber|innerhalb|außerhalb|oberhalb|unterhalb"
                r"|während|wegen|trotz))\s+"
                r"|([A-ZÄÖÜ][a-zäöüß]+)\s+)?"
                r"\b([Aa]m|[Ii]m|[Vv]om|[Zz]um|[Zz]ur|[Bb]eim|[Dd]em|[Dd]en|[Dd]es"
                r"|[Dd]er|[Dd]ie|[Dd]as)\s+"
                r"(\d{1,2})\.\s+"
                r"(?!(?:Das|Der|Die|Den|Dem|Des|Dann|Danach|Davor|Daher|Darum"
                r"|Deshalb|Dies|Diese|Dieser|Dieses|Er|Es|Wir|Ihr|Ich|Du|Man"
                r"|Aber|Und|Oder|Doch|Denn|Nun|Jetzt|Heute|Morgen|Gestern"
                r"|Bitte|Danke|Was|Wer|Wie|Wo|Warum|Wann|Auch|Alle|Alles|So"
                r"|Dort|Hier|Ja|Nein|Kein|Keine|Mein|Meine|Sein|Seine|Dein"
                r"|Deine|Unser|Unsere|Ihre|Als|Wenn|Ob|Leider|Nach|Vor|Bei"
                r"|Mit|Von|Zum|Zur|Am|Im|In|An|Auf|Für|Um|Zu|Aus|Damit"
                r"|Trotzdem|Außerdem|Vielleicht|Natürlich|Sie)\b)"
                r"([A-ZÄÖÜ][a-zäöüß]+)",
                _ord_prose, out)

        # units: number + unit
        unit_map = _UNITS["de" if de else "en"]
        unit_alt = "|".join(re.escape(u) for u in sorted(unit_map, key=lambda x: -len(x)))
        def _unit(m):
            spoken = _spell_amount(m.group(1), lang)
            return f"{spoken} {unit_map[m.group(2)]}"
        out = re.sub(rf"({amt})\s?({unit_alt})\b", _unit, out)

        # percentages before the generic number rule
        def _pct(m):
            unit = "Prozent" if de else "percent"
            return f"{_spell_amount(m.group(1), lang)} {unit}"
        out = re.sub(rf"({amt})\s?%", _pct, out)

        # year ranges ("1914-1918", also en/em dash) read as "X bis Y"/"X to Y"
        def _yrange(m):
            joiner = "bis" if de else "to"
            return (f"{year_to_words(int(m.group(1)), lang)} {joiner} "
                    f"{year_to_words(int(m.group(2)), lang)}")
        out = re.sub(
            r"(?<![\d.,'])((?:1[1-9]|20)\d{2})\s?[–—-]\s?((?:1[1-9]|20)\d{2})(?![.,]?\d)",
            _yrange, out)

        # standalone 4-digit years (1100-2099) BEFORE the generic number rule,
        # so "1920" reads as a year, not "one thousand nine hundred twenty".
        # Decade suffixes keep working: "1920er"/"1920ern" (DE), "1990s" (EN).
        def _year(m):
            words = year_to_words(int(m.group(1)), lang)
            suffix = m.group(2) or ""
            if suffix == "s":
                return words[:-1] + "ies" if words.endswith("y") else words + "s"
            return words + suffix
        out = re.sub(
            r"(?<![A-Za-z\d.,'])((?:1[1-9]|20)\d{2})(ern|er|s)?(?![.,]?\d)\b",
            _year, out)

        # any remaining standalone numbers / decimals
        def _num(m):
            return _spell_amount(m.group(0), lang)
        out = re.sub(r"\d[\d'.,]*\d|\d", _num, out)

        out = out.replace("&", " und " if de else " and ")

        return re.sub(r"\s{2,}", " ", out).strip()
    except Exception:  # noqa: BLE001 - never break synthesis
        return text


def _parse_number(token: str, lang: str) -> tuple[str, str]:
    """
    Split a raw numeric token into (integer_digits, fraction_digits),
    disambiguating thousands/decimal separators.
    Apostrophes are always thousands (Swiss: 1'234.50). When both
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
    return re.sub(r"\D", "", int_str), re.sub(r"\D", "", frac)


def _spell_amount(token: str, lang: str) -> str:
    """Turn a raw numeric token into words ("3.5" -> "drei Komma fünf").

    A leading zero on a whole number ("016", "007") is read digit-by-digit,
    not as a cardinal — int() would otherwise silently drop it ("016" ->
    "sechzehn"/"sixteen"), which is wrong for the identifiers a leading zero
    always signals (order/room/reference numbers, phone-number groups: a
    real quantity is never written with one). Matches real shipping TTS
    normalizers (NVIDIA NeMo, Ivona): a digit run is classified before
    verbalization, and cardinal-vs-identifier is a leading-zero/shape
    decision, not something a value-preserving int() conversion can make.
    Space-separated phone-number groups without a leading zero ("34", "56"
    in "016 34 56") are tokenized separately already and correctly keep
    their normal cardinal reading — that already matches how German phone
    numbers are conventionally grouped and read (DIN 5008).
    """
    int_str, frac = _parse_number(token, lang)
    if not int_str:
        return token
    if not frac and len(int_str) > 1 and int_str[0] == "0":
        return digits_spelled(int_str, lang)
    if frac:
        return _decimal_to_words(int(int_str), frac, lang)
    return number_to_words(int(int_str), lang)
