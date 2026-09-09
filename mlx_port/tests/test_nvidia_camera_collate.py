"""NVIDIA camera-order / collation / conversation image contracts.

Weight-free. Needs torch (macos-26 mlx-ci). Complementary to open #17–#24,
which covered label masks and train-vs-gen prompts — not camera sort or
collate stacking.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_SPECIAL_TOKEN_KEYS = [
    "traj_history_start",
    "traj_history_end",
    "traj_history",
    "traj_future_start",
    "traj_future_end",
    "traj_future",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
]


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _ensure_base_model_specials() -> None:
    _ensure_src_on_path()
    if "alpamayo_r1.models.base_model" in sys.modules:
        return
    try:
        import alpamayo_r1.models.base_model  # noqa: F401

        return
    except Exception:
        sys.modules.pop("alpamayo_r1.models.base_model", None)
    import alpamayo_r1  # noqa: F401

    models = ModuleType("alpamayo_r1.models")
    models.__path__ = [str(_SRC / "alpamayo_r1" / "models")]
    sys.modules["alpamayo_r1.models"] = models
    sys.modules["alpamayo_r1.models.base_model"] = SimpleNamespace(
        SPECIAL_TOKENS={k: f"<|{k}|>" for k in _SPECIAL_TOKEN_KEYS},
        IGNORE_INDEX=-100,
    )


def _load_fns(rel: str, names: tuple[str, ...], extra: dict | None = None):
    path = _SRC / rel
    tree = ast.parse(path.read_text())
    keep = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    missing = set(names) - {node.name for node in keep}
    if missing:
        raise AssertionError(f"{path} is missing {sorted(missing)}")
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"__name__": "extracted", **(extra or {})}
    exec(compile(mod, str(path), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_processor_fns(torch):
    _ensure_src_on_path()
    try:
        return _load_fns(
            "alpamayo_r1/processor/qwen_processor.py",
            ("sort_images_by_camera_ids", "basic_collation_fn"),
            extra={"torch": torch},
        )
    except Exception as exc:
        pytest.skip(f"NVIDIA processor helpers unavailable: {exc}")


def _load_conversation():
    _ensure_src_on_path()
    _ensure_base_model_specials()
    path = _SRC / "alpamayo_r1" / "chat_template" / "conversation.py"
    spec = importlib.util.spec_from_file_location("nv_conversation_cam", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nv_conversation_cam"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA conversation unavailable: {exc}")
    return module


def _load_helper_create_message():
    _ensure_src_on_path()
    try:
        return _load_fns("alpamayo_r1/helper.py", ("create_message",))["create_message"]
    except Exception as exc:
        pytest.skip(f"NVIDIA helper.create_message unavailable: {exc}")


def test_sort_images_by_camera_ids_is_stable_and_reorders_timestamps(torch):
    fns = _load_processor_fns(torch)
    # Duplicate camera 1 must keep input order (stable argsort).
    frames = torch.arange(4 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 2, 3, 2, 2)
    cams = torch.tensor([2, 1, 1, 0], dtype=torch.long)
    stamps = torch.tensor([20.0, 11.0, 12.0, 0.0])
    sorted_frames, sorted_cams, sorted_stamps = fns["sort_images_by_camera_ids"](
        frames, cams, relative_timestamps=stamps, return_camera_ids=True
    )
    assert sorted_cams.tolist() == [0, 1, 1, 2]
    assert torch.equal(sorted_frames[0], frames[3])
    assert torch.equal(sorted_frames[1], frames[1])
    assert torch.equal(sorted_frames[2], frames[2])
    assert torch.equal(sorted_frames[3], frames[0])
    assert sorted_stamps.tolist() == [0.0, 11.0, 12.0, 20.0]

    images_only = fns["sort_images_by_camera_ids"](frames, cams)
    assert torch.equal(images_only, sorted_frames)


def test_basic_collation_stacks_tensors_and_keeps_unstackable_lists(torch):
    fns = _load_processor_fns(torch)
    batch = [
        {"ids": torch.tensor([1, 2]), "clip_id": "a", "meta": torch.tensor([0.5])},
        {"ids": torch.tensor([3, 4]), "clip_id": "b", "meta": torch.tensor([1.5])},
    ]
    out = fns["basic_collation_fn"](batch, unstackable_keys=["meta"])
    assert out["ids"].shape == (2, 2)
    assert out["ids"].tolist() == [[1, 2], [3, 4]]
    assert out["clip_id"] == ["a", "b"]
    assert out["meta"] == [batch[0]["meta"], batch[1]["meta"]]
    assert isinstance(out["meta"], list)


def test_construct_image_requires_sorted_cameras_and_labels_views(torch):
    conv = _load_conversation()
    frames = torch.zeros((2, 2, 3, 4, 4))
    frames[0] = 1.0
    frames[1] = 2.0
    try:
        conv.construct_image(
            {"image_frames": frames},
            include_camera_ids=True,
            camera_ids=torch.tensor([1, 0]),
            include_frame_nums=False,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError when camera ids are unsorted")

    msgs = conv.construct_image(
        {"image_frames": frames},
        include_camera_ids=True,
        camera_ids=torch.tensor([0, 1]),
        include_frame_nums=True,
    )
    kinds = [m["type"] for m in msgs]
    texts = [m.get("text") for m in msgs if m["type"] == "text"]
    assert kinds == ["text", "text", "image", "text", "image", "text", "text", "image", "text", "image"]
    assert texts[0] == "Front left camera: "
    assert texts[1] == "frame 0 "
    assert texts[3] == "Front camera: "
    assert sum(1 for k in kinds if k == "image") == 4


def test_construct_traj_history_and_meta_action_guards():
    conv = _load_conversation()
    hist = conv.construct_traj_history(2)
    assert hist[0]["text"] == (
        "<|traj_history_start|><|traj_history|><|traj_history|><|traj_history_end|>"
    )
    try:
        conv.construct_meta_action({}, ask_for_component=False)
    except AssertionError as exc:
        assert "meta_action" in str(exc)
    else:
        raise AssertionError("expected AssertionError when teacher meta_action is missing")
    ask = conv.construct_meta_action({}, ask_for_component=True)
    assert ask[0]["text"] == "<|meta_action_start|>"
    filled = conv.construct_meta_action(
        {"meta_action_strings": "lane keep"}, ask_for_component=False
    )
    assert filled[0]["text"] == "<|meta_action_start|>lane keep<|meta_action_end|>"


def test_nvidia_helper_create_message_requires_nchw_and_pads_48_hist(torch):
    create_message = _load_helper_create_message()
    try:
        create_message(torch.zeros((3, 8, 8)))
    except ValueError as exc:
        assert "expected 4" in str(exc)
    else:
        raise AssertionError("expected ValueError for rank-3 frames")
    msgs = create_message(torch.zeros((2, 3, 8, 8)))
    user = "".join(part["text"] for part in msgs[1]["content"] if part["type"] == "text")
    assert user.count("<|traj_history|>") == 48
    assert user.startswith("<|traj_history_start|>")
    assert msgs[2]["content"][0]["text"] == "<|cot_start|>"
    assert sum(1 for part in msgs[1]["content"] if part["type"] == "image") == 2
