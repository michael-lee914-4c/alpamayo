"""PAI ``get_clip_feature`` raises on unexpected extensions.

Weight-free. Complementary to open #33 (unknown feature → ``None``, not
raise). A ``.bin`` / ``.json`` shard is a hard error so a renamed parquet
is not silently treated as a zip of cameras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PAI_UTILS = _REPO / "src" / "alpamayo_r1" / "data" / "pai_utils.py"


def test_source_unexpected_extension_raises():
    src = _PAI_UTILS.read_text()
    assert 'if chunk_filename.endswith(".parquet"):' in src
    assert 'elif chunk_filename.endswith(".zip"):' in src
    assert 'raise ValueError(f"Unexpected file extension: {chunk_filename=}.")' in src
    assert "if feature not in self.features.features_df.index:" in src
    assert "return None" in src


def test_unexpected_extension_is_value_error_not_none():
    def _read_by_ext(chunk_filename: str, feature_in_index: bool):
        if not feature_in_index:
            return None
        if chunk_filename.endswith(".parquet"):
            return "parquet"
        if chunk_filename.endswith(".zip"):
            return "zip"
        raise ValueError(f"Unexpected file extension: {chunk_filename=}.")

    assert _read_by_ext("chunk/foo.parquet", True) == "parquet"
    assert _read_by_ext("chunk/cam.zip", True) == "zip"
    assert _read_by_ext("chunk/foo.bin", False) is None
    with pytest.raises(ValueError, match="Unexpected file extension"):
        _read_by_ext("chunk/foo.bin", True)
    with pytest.raises(ValueError, match="Unexpected file extension"):
        _read_by_ext("chunk/foo.json", True)
