"""
Cross-source player name matching utility.
Player names differ across NBA API, Odds API, Kalshi, and Polymarket.
"""

import re
from difflib import SequenceMatcher


# Common suffixes to strip when normalizing
_SUFFIXES = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", re.IGNORECASE)
_EXTRA_SPACES = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Lowercase, strip suffixes (Jr., III), standardize spacing."""
    if not name:
        return ""
    name = name.strip().lower()
    name = _SUFFIXES.sub("", name)
    name = _EXTRA_SPACES.sub(" ", name)
    # Remove periods (e.g. "L. James" -> "L James")
    name = name.replace(".", "")
    return name.strip()


def fuzzy_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """Return True if names are close enough to be the same person."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if n1 == n2:
        return True
    ratio = SequenceMatcher(None, n1, n2).ratio()
    return ratio >= threshold


def find_best_match(target_name: str, candidates: list[str], threshold: float = 0.85) -> str | None:
    """
    Find best matching name from a list of candidates.
    Returns None if no match exceeds the threshold.
    """
    target_norm = normalize_name(target_name)
    best_score = 0.0
    best_match = None

    for candidate in candidates:
        cand_norm = normalize_name(candidate)
        if target_norm == cand_norm:
            return candidate
        score = SequenceMatcher(None, target_norm, cand_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return best_match
    return None


def build_name_map(source_names: list[str], target_names: list[str], threshold: float = 0.85) -> dict[str, str]:
    """
    Build a mapping from source names to target names.
    Returns {source_name: best_target_name} for matched pairs.
    """
    mapping = {}
    for name in source_names:
        match = find_best_match(name, target_names, threshold)
        if match:
            mapping[name] = match
    return mapping
