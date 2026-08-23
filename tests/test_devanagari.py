# -*- coding: utf-8 -*-
"""Devanagari->Roman transliteration contract tests."""

from utils.devanagari import has_devanagari, to_roman


def test_latin_text_passes_through():
    assert to_roman("Pant OUT hote hi pudiya band!") == "Pant OUT hote hi pudiya band!"


def test_has_devanagari_detection():
    assert has_devanagari("ये छक्का था")
    assert not has_devanagari("ye chhakka tha")


def test_cricket_vocabulary_becomes_searchable():
    roman = to_roman("छक्का विकेट चौका शतक")
    assert "chhaka" in roman or "chhakkaa" in roman
    assert "vike" in roman          # विकेट -> vikeTa/viket
    assert "shatak" in roman


def test_player_names_transliterate():
    roman = to_roman("विराट कोहली ने बुमराह को छक्का मारा")
    assert "viraa" in roman.replace(" ", "")[:8] or "viraat" in roman
    assert "bumaraa" in roman or "bumrah" in roman


def test_mixed_script_line():
    roman = to_roman("ये DEBATE बेकार है")
    assert "DEBATE" in roman            # latin preserved verbatim
    assert not has_devanagari(roman)


def test_cluster_kya_reads_like_hinglish():
    roman = to_roman("क्या बात है")
    assert roman.startswith("ky")


def test_empty_and_none_safe():
    assert to_roman("") == ""
    assert to_roman(None) == ""
