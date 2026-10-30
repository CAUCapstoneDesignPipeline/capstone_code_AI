"""Exercise the canonical image's project installer against a stale local wheel.

The tiny PEP517 fixture backend uses only stdlib, so no registry or model is needed.
The pyproject and backend stay byte-identical while only package source changes.
"""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = '''import base64, csv, hashlib, io, pathlib, zipfile

def get_requires_for_build_wheel(config_settings=None):
    return []

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    files = {str(p): p.read_bytes() for p in pathlib.Path("app").glob("*.py")}
    files["ai-0.1.0.dist-info/METADATA"] = b"Metadata-Version: 2.1\\nName: ai\\nVersion: 0.1.0\\n"
    files["ai-0.1.0.dist-info/WHEEL"] = b"Wheel-Version: 1.0\\nGenerator: regression-fixture\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n"
    record = io.StringIO()
    writer = csv.writer(record)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow(["ai-0.1.0.dist-info/RECORD", "", ""])
    files["ai-0.1.0.dist-info/RECORD"] = record.getvalue().encode()
    name = "ai-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(pathlib.Path(wheel_directory) / name, "w") as wheel:
        for path, data in files.items():
            wheel.writestr(path, data)
    return name
'''


def test_image_installer_rebuilds_changed_app_with_unchanged_project_metadata(tmp_path):
    uv = os.environ.get("UV_BIN") or shutil.which("uv")
    if not uv and (ROOT / ".local/tools/bin/uv").exists():
        uv = str(ROOT / ".local/tools/bin/uv")
    if not uv:
        pytest.skip("uv executable required for real installer cache regression")
    uv = str(Path(uv).resolve())
    project = tmp_path / "project"
    project.mkdir()
    (project / "app").mkdir()
    (project / "app/__init__.py").write_text("")
    source = project / "app/runtime.py"
    source.write_text("VALUE = 'first-source'\n")
    pyproject = project / "pyproject.toml"
    pyproject.write_text('''[project]
name = "ai"
version = "0.1.0"
requires-python = ">=3.11"
[build-system]
requires = []
build-backend = "backend"
backend-path = ["."]
''')
    backend = project / "backend.py"
    backend.write_text(BACKEND)
    original_metadata = (pyproject.read_bytes(), backend.read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PYTHONPATH"}}
    env.update(UV_CACHE_DIR=str(tmp_path / "uv-cache"), UV_PYTHON_DOWNLOADS="never",
               UV_OFFLINE="1", UV_NO_PROGRESS="1", PYTHONDONTWRITEBYTECODE="1")

    def run(args, cwd=project):
        return subprocess.run(args, cwd=cwd, env=env, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60).stdout.strip()

    run([uv, "lock", "--python", sys.executable])
    # Use the actual production command, without the fixture's nonexistent embedding extra.
    line = next(x.strip() for x in (ROOT / "Dockerfile").read_text().splitlines()
                if "uv sync" in x and "--no-editable" in x)
    installer = [uv, *[x for x in shlex.split(line)[1:] if x != "${UV_EXTRAS}"]]
    run([*installer, "--python", sys.executable])
    installed_python = str(project / ".venv/bin/python")

    def installed_value():
        # Outside project cwd: importing checkout source would conceal stale wheel installs.
        return run([installed_python, "-c", "import app.runtime; print(app.runtime.VALUE)"], cwd=tmp_path)

    assert installed_value() == "first-source"
    source.write_text("VALUE = 'second-source'\n")
    assert original_metadata == (pyproject.read_bytes(), backend.read_bytes())
    # A new Docker project layer starts without an installed project but retains wheel cache.
    run([uv, "pip", "uninstall", "--python", installed_python, "ai"])
    run([*installer, "--python", sys.executable])
    assert installed_value() == "second-source"
