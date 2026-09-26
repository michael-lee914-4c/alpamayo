"""PAI ``chunk_ids`` hyphen strings are end-exclusive ``range(start, end)``.

Weight-free. Complementary to open #42 (``int → [int]``) and #33 (invalid
float / comma-str character-split). ``\"0-3\"`` becomes ``[0, 1, 2]`` — chunk
3 is dropped. Operators who read the range as inclusive would silently
omit a whole shard from train/eval.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PAI = _REPO / "src" / "alpamayo_r1" / "data" / "pai_utils.py"


def _parse_chunk_ids(chunk_ids):
    if isinstance(chunk_ids, str) and "-" in chunk_ids:
        chunk_start = int(chunk_ids.split("-")[0])
        chunk_end = int(chunk_ids.split("-")[1])
        return list(range(chunk_start, chunk_end))
    if isinstance(chunk_ids, int):
        return [chunk_ids]
    return list(chunk_ids)


def test_source_uses_end_exclusive_range():
    src = _PAI.read_text()
    assert 'isinstance(chunk_ids, str) and "-" in chunk_ids' in src
    assert "self.chunk_ids = list(range(chunk_start, chunk_end))" in src
    assert "range(chunk_start, chunk_end + 1)" not in src


def test_hyphen_range_drops_the_end_id():
    assert _parse_chunk_ids("0-3") == [0, 1, 2]
    assert 3 not in _parse_chunk_ids("0-3")
    assert _parse_chunk_ids("0-9") == list(range(0, 9))


def test_int_and_list_paths_are_unchanged():
    assert _parse_chunk_ids(4) == [4]
    assert _parse_chunk_ids([1, 3, 5]) == [1, 3, 5]
