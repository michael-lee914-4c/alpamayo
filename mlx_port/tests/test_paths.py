"""Layout of mlx_port/reports and mlx_port/doc."""

from mlx_port.paths import DOC_DIR, MLX_PORT_DIR, REPORTS_DIR, REPO_ROOT


def test_reports_live_under_mlx_port():
    assert MLX_PORT_DIR.name == "mlx_port"
    assert REPORTS_DIR == MLX_PORT_DIR / "reports"
    assert REPORTS_DIR.is_dir()
    assert (REPORTS_DIR / "stage1c_progress.html").is_file()
    assert (REPORTS_DIR / "stage_template.html").is_file()
    assert not (REPO_ROOT / "reports").exists()


def test_train_how_to_lives_under_doc():
    assert DOC_DIR == MLX_PORT_DIR / "doc"
    assert (DOC_DIR / "train_lora_vs_dense.html").is_file()
    assert (DOC_DIR / "train_lora_vs_dense" / "train-arch.png").is_file()
    assert not (REPORTS_DIR / "train_lora_vs_dense.html").exists()
