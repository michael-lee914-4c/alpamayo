"""Repo-relative locations for the MLX port."""

from pathlib import Path

MLX_PORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = MLX_PORT_DIR.parent
REPORTS_DIR = MLX_PORT_DIR / "reports"
DOC_DIR = MLX_PORT_DIR / "doc"
