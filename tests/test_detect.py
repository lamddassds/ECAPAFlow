"""Language-detection tests for the Auto mode — runnable standalone:

    python tests/test_detect.py

Importing app is safe here: the engine loads lazily via the FastAPI
lifespan, which never runs on plain import.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402

CASES = [
    ("Der Krieg begann 1914 und endete 1918 mit vielen Opfern.", "de"),
    ("Schon 1920 war alles ganz anders als zuvor.", "de"),
    ("The war started in 1914 and ended right after 1918.", "en"),
    ("The roaring 1920s started right after 1918.", "en"),
    ("La guerra comenzó en 1914 y terminó con muchas víctimas.", "es"),
    ("La guerre a commencé en 1914 et s'est terminée dans les tranchées.", "fr"),
    ("Il conflitto è iniziato nel 1914 e si è concluso con la vittoria.", "it"),
    ("De oorlog begon in 1914 en het einde kwam pas veel later.", "nl"),
    ("전쟁은 1914년에 시작되었다.", "ko"),
    ("戦争は1914年に始まりました。", "ja"),
    ("Война началась в 1914 году.", "ru"),
    ("الحرب بدأت في عام ١٩١٤", "ar"),
    ("xyz qqq 123", "na"),          # nothing to go on -> engine auto
    ("", "na"),
    # casual/conversational German: confirmed bug, 2026-07-05 — scored zero
    # hits before conversational words were added, misdetecting as "na" and
    # defeating Auto Voice for exactly this kind of everyday greeting.
    ("Hallo, wie geht es dir heute?", "de"),
    ("Guten Morgen, danke für die Nachricht.", "de"),
]


def test_detection():
    for text, expected in CASES:
        got = app._detect_lang(text)
        assert got == expected, f"{text[:40]!r}: expected {expected}, got {got}"


if __name__ == "__main__":
    try:
        test_detection()
        print(f"PASS test_detection ({len(CASES)} cases)")
    except AssertionError as e:
        print(f"FAIL test_detection: {e}")
        sys.exit(1)
