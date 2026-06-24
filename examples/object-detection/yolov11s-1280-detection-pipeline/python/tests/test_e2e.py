"""E2E tests for yolov11s-1280-detection-pipeline (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"
LABELS_FILE = EXAMPLE_DIR / "common" / "coco_label.txt"


def _find_model(models_dir: Path) -> Path | None:
    if not models_dir.exists():
        return None
    for f in models_dir.iterdir():
        name = f.name
        if (("yolo11s" in name or "yolov11s" in name or "yolo_v11s" in name
                or "yolo11" in name or name == "1280.tar.gz")
                and "seg" not in name and name.endswith(".tar.gz")):
            return f
    return None


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline_folder(
        self, models_dir, tmp_output_dir, test_images_dir, test_timeout_ms, skip_unless_e2e_ready
    ):
        model = _find_model(models_dir)
        skip_unless_e2e_ready(model is not None, "yolo11s model (.tar.gz) not found in models_dir")
        skip_unless_e2e_ready(LABELS_FILE.exists(), f"Labels file not found: {LABELS_FILE}")
        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            "test_images_dir is missing or empty",
        )

        result = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                str(model),
                str(LABELS_FILE),
                str(test_images_dir),
                str(tmp_output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        output_files = list(tmp_output_dir.iterdir())
        assert len(output_files) > 0, "Expected output files but output directory is empty"

    def test_full_pipeline_rtsp(
        self, models_dir, tmp_output_dir, test_timeout_ms, skip_unless_e2e_ready
    ):
        import os
        rtsp_url = os.environ.get("SIMANEAT_APPS_TEST_RTSP_URL", "").strip()
        skip_unless_e2e_ready(
            bool(rtsp_url) and not rtsp_url.startswith("<"),
            "set SIMANEAT_APPS_TEST_RTSP_URL to a valid RTSP URL to run RTSP e2e test",
        )

        model = _find_model(models_dir)
        skip_unless_e2e_ready(model is not None, "yolo11s model (.tar.gz) not found in models_dir")
        skip_unless_e2e_ready(LABELS_FILE.exists(), f"Labels file not found: {LABELS_FILE}")

        result = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                str(model),
                str(LABELS_FILE),
                str(tmp_output_dir),
                "--rtsp", rtsp_url,
                "--frames", "10",
            ],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py (RTSP) exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        output_files = list(tmp_output_dir.iterdir())
        assert len(output_files) > 0, "Expected output frames but output directory is empty"
