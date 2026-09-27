"""PAI loader ``argsort`` is unstable; Qwen processor sort is ``stable=True``.

Weight-free. Complementary to open #25 (camera-id stable sort) and #41
(sort permutes ``relative_timestamps``). ``load_physical_aiavdataset``
uses ``torch.argsort(camera_indices)`` with the default unstable
algorithm, then derives ``relative_timestamps`` from the *sorted*
``all_timestamps.min()``. Tied camera IDs can flip frame order (and
therefore tmin) on the loader path but not on the processor path.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_LOAD = _REPO / "src" / "alpamayo_r1" / "load_physical_aiavdataset.py"
_QWEN = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def test_source_loader_argsort_has_no_stable_flag():
    src = _LOAD.read_text()
    assert "sort_order = torch.argsort(camera_indices)" in src
    assert "torch.argsort(camera_indices, stable=True)" not in src


def test_source_processor_sort_is_stable():
    src = _QWEN.read_text()
    body = src.split("def sort_images_by_camera_ids")[1].split("def basic_collation_fn")[0]
    assert "sorted_indices = torch.argsort(camera_indices, stable=True)" in body


def test_unstable_tie_can_flip_equal_ids():
    # document the contract: equal IDs are a defined processor-stable
    # order (original) and an undefined loader order
    ids = [1, 1, 2]
    # stable sort of a tie keeps original relative order
    stable = sorted(range(len(ids)), key=lambda i: (ids[i], i))
    assert [ids[i] for i in stable] == [1, 1, 2]
    assert stable[:2] == [0, 1]
