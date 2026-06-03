from __future__ import annotations

import re


_STOPWORDS = {"the", "a", "an", "to", "of", "and", "or", "can", "could"}

_TITLE_ALIAS_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("黑客帝国3", "黑客帝国iii", "矩阵革命", "矩陣革命", "骇客任务完结篇", "駭客任務完結篇"),
        ("matrix revolutions", "the matrix revolutions"),
    ),
    (
        ("黑客帝国2", "黑客帝国ii", "矩阵重装上阵", "矩陣重裝上陣", "骇客任务重装上阵", "駭客任務重裝上陣"),
        ("matrix reloaded", "the matrix reloaded"),
    ),
    (
        ("黑客帝国4", "黑客帝国iv", "矩阵复活", "矩陣復活", "骇客任务复活", "駭客任務復活"),
        ("matrix resurrections", "the matrix resurrections"),
    ),
    (
        ("黑客帝国", "骇客任务", "駭客任務", "矩阵系列", "矩陣系列"),
        ("matrix", "the matrix"),
    ),
)


def title_query_variants(value: str) -> list[str]:
    variants: list[str] = []

    def add(candidate: str) -> None:
        normalized = normalize_title_text(candidate)
        if normalized and normalized not in variants:
            variants.append(normalized)

    add(value)
    lowered = value.casefold()
    compact = re.sub(r"\s+", "", lowered)
    for aliases, expansions in _TITLE_ALIAS_GROUPS:
        if any(alias.casefold() in lowered or alias.casefold() in compact for alias in aliases):
            for expansion in expansions:
                add(expansion)
    return variants


def title_tokens_from_text(value: str) -> list[str]:
    best: list[str] = []
    for variant in title_query_variants(value):
        tokens = [
            token
            for token in normalize_title_text(variant).split()
            if token not in _STOPWORDS
        ]
        if len(tokens) > len(best) or (len(tokens) == len(best) and len(" ".join(tokens)) > len(" ".join(best))):
            best = tokens
    return best


def normalize_title_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
