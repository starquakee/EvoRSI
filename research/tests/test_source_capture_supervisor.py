"""Source checking and admission must use the same immutable bytes."""
import os
import pytest
from research.contracts.source_gate import load_policy, check_source_tree
from research.contracts.source_snapshot import CaptureError, capture_source

def test_symlink_subdirectory_rejected(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "main.py").write_text("import extra.payload\n")
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "payload.py").write_text("import subprocess\n")
    (code / "extra").symlink_to(outside, target_is_directory=True)
    verdict = check_source_tree(code, load_policy())
    assert not verdict.allowed
    assert any(item.rule == "symlink_denied" for item in verdict.denials)

def test_capture_materializes_original_bytes_and_preserves_upload(tmp_path):
    upload = tmp_path / "main.py"
    upload.write_bytes(b"print(1)\n")
    snapshot = capture_source(upload, load_policy(), allowed_root=tmp_path)
    upload.write_bytes(b"import subprocess\n")
    job = tmp_path / "job"
    job.mkdir()
    saved = snapshot.materialize(job)
    assert saved.read_bytes() == b"print(1)\n"
    assert upload.exists()
    assert check_source_tree(saved, load_policy()).source_sha256 == snapshot.verdict.source_sha256

def test_assets_are_bound_to_identity(tmp_path):
    (tmp_path / "main.py").write_text("print(1)\n")
    asset = tmp_path / "data.txt"
    asset.write_text("a")
    first = capture_source(tmp_path, load_policy())
    asset.write_text("b")
    second = capture_source(tmp_path, load_policy())
    assert first.verdict.source_sha256 != second.verdict.source_sha256

def test_source_fifo_rejected_without_blocking(tmp_path):
    path = tmp_path / "main.py"
    os.mkfifo(path)
    with pytest.raises(CaptureError, match="regular"):
        capture_source(path, load_policy())

def test_outside_allowed_storage_rejected(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    source = tmp_path / "main.py"
    source.write_text("print(1)\n")
    with pytest.raises(CaptureError, match="escapes"):
        capture_source(source, load_policy(), allowed_root=storage)

def test_linked_parent_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "main.py").write_text("print(1)\n")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(CaptureError) as error:
        capture_source(link / "main.py", load_policy())
    assert error.value.rule == "symlink_denied"
