"""높은 재현율 후보 생성(토큰 n-gram): 영어 구절을 만들고, 문장부호·조사·가린 구간에서 끊는다."""

import argparse
from pathlib import Path

import pytest

from app.cli import load_extractor
from experiments.distill.candidates import generate

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ex():
    return load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None)) if (ROOT / "data/private/concepts/idf.json").exists() \
        else __import__("app.services.concepts", fromlist=["ConceptExtractor"]).ConceptExtractor()


def keys(ex, text):
    return set(generate(ex, text)[1])


def test_english_noun_phrases_and_their_sub_phrases_are_candidates(ex):
    k = keys(ex, "A time-dependent vector field defines the Flow Matching objective for generative modeling.")
    assert {"vector field", "flow matching", "generative modeling"} <= k


def test_candidates_do_not_cross_sentence_punctuation_or_lines(ex):
    k = keys(ex, "Diffusion models work well. Transformers scale\nGaussian noise")
    assert not any("well transformers" in c or "scale gaussian" in c or "." in c for c in k)      # 문장부호·줄바꿈을 넘지 않는다
    assert not any(c.endswith(" w") or c.startswith("ell") for c in k)                           # 낱말 조각이 아니다
    assert {"diffusion models", "transformers", "gaussian noise"} <= k


def test_korean_stems_before_particles_and_compound_nouns(ex):
    k = keys(ex, "체크포인트는 복구 시간을 줄인다. 페이지 캐시를 쓴다.")
    assert {"체크포인트", "페이지 캐시"} <= k


def test_inline_code_and_math_are_not_candidates(ex):
    k = keys(ex, "함수 `secret_token` 은 $x_1$ 를 쓴다.")
    assert not any("secret" in c or "x_1" in c for c in k)


def test_spans_point_at_the_original_text(ex):
    text, cands = generate(ex, "Vision Transformer 는 패치를 토큰으로 본다.")
    for c in cands.values():
        for s, e in c.spans:
            assert text[s:e].lower().replace("  ", " ") and len(text[s:e]) <= 60
