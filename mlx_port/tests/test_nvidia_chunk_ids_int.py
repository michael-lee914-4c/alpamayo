"""A lone ``chunk_ids`` int becomes ``[int]``; range strings stay end-exclusive.

Weight-free. Complementary to open #27 (``"0-9"`` → ``range(0, 9)``) and
#33 (float / comma-str → None / character-split). Passing ``3`` as an int
must not fall through to the invalid branch and load every chunk.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PAI = _REPO / "src" / "alpamayo_r1" / "data" / "pai_utils.py"


def _parse_chunk_ids(chunk_ids):
    if chunk_ids is None:
        return None
    if isinstance(chunk_ids, str) and "-" in chunk_ids:
        chunk_start = int(chunk_ids.split("-")[0])
        chunk_end = int(chunk_ids.split("-")[1])
        return list(range(chunk_start, chunk_end))
    if isinstance(chunk_ids, (list, tuple, Iterable)) and not isinstance(chunk_ids, (str, bytes)):
        return list(chunk_ids)
    if isinstance(chunk_ids, int):
        return [chunk_ids]
    return None


def test_source_int_is_its_own_branch_after_iterable():
    src = _PAI.read_text()
    assert "elif isinstance(chunk_ids, int):" in src
    assert "self.chunk_ids = [chunk_ids]" in src
    # Int must not be swallowed by the Iterable branch (int is not Iterable).
    assert "elif isinstance(chunk_ids, (list, tuple, Iterable)):" in src
    assert "self.chunk_ids = list(range(chunk_start, chunk_end))" in src


def test_single_int_is_a_one_element_list():
    assert _parse_chunk_ids(3) == [3]
    assert _parse_chunk_ids(0) == [0]
    assert _parse_chunk_ids(None) is None


def test_int_is_not_the_same_as_a_range_string():
    assert _parse_chunk_ids("0-3") == [0, 1, 2]
    assert _parse_chunk_ids(3) == [3]
    assert _parse_chunk_ids([3]) == [3]
    # A float is not an int and is not a range string.
    assert _parse_chunk_ids(3.0) is None
