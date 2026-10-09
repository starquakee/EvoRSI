"""Contract tests for research/tools/audit_secrets.py (US-001).

Planted secret samples are assembled at runtime from fragments so this test
file itself stays clean under the repository-wide audit.
"""
from __future__ import annotations

from pathlib import Path

from research.tools.audit_secrets import (
    audit,
    scan_text_for_host_paths,
    scan_text_for_secrets,
)


def _sandbox_key() -> str:
    return "mlsandbox-" + "a1b2c3d4e5f60718"


def _generic_assignment() -> str:
    return "API_TOKEN = \"" + "t" * 24 + "\""


class TestSecretScan:
    def test_sandbox_key_detected(self):
        findings = scan_text_for_secrets(f'api_key = "{_sandbox_key()}"\n')
        patterns = [f["pattern"] for f in findings]
        assert "openmle_sandbox_key" in patterns
        assert "generic_secret_assignment" in patterns
        assert findings[0]["line"] == 1

    def test_generic_assignment_detected_on_correct_line(self):
        findings = scan_text_for_secrets(f"# comment\n{_generic_assignment()}\n")
        assert any(f["pattern"] == "generic_secret_assignment" and f["line"] == 2 for f in findings)

    def test_private_key_block_detected(self):
        header = "-----BEGIN " + "RSA PRIVATE KEY-----"
        assert any(f["pattern"] == "private_key_block" for f in scan_text_for_secrets(header))

    def test_placeholder_values_ignored(self):
        assert scan_text_for_secrets('API_KEY = "<your-key-here>"\n') == []

    def test_env_reads_are_not_secrets(self):
        clean = (
            "import os\n"
            'api_key = os.environ["SANDBOX_API_KEY"]\n'
            'headers = {"X-API-Key": api_key}\n'
        )
        assert scan_text_for_secrets(clean) == []

    def test_clean_ml_code_passes(self):
        assert scan_text_for_secrets("import pandas as pd\ndf = pd.read_csv(p)\n") == []


class TestHostPathScan:
    def test_linux_home_detected(self):
        findings = scan_text_for_host_paths("p = \"/home/someone/x\"\n")
        assert [f["pattern"] for f in findings] == ["linux_home"]

    def test_wsl_drive_detected(self):
        findings = scan_text_for_host_paths("root = \"/mnt/c/Users/x\"\n")
        patterns = {f["pattern"] for f in findings}
        assert "wsl_windows_drive" in patterns

    def test_windows_path_detected(self):
        findings = scan_text_for_host_paths('base = "C:\\Users\\x"\n')
        assert any(f["pattern"] == "windows_user_dir" for f in findings)

    def test_relative_and_worker_paths_allowed(self):
        clean = 'root = Path(__file__).resolve().parents[2]\ndata = "/mnt/pubdatasets2/tasks"\n'
        assert scan_text_for_host_paths(clean) == []


class TestAuditPolicy:
    def test_host_path_hard_fail_only_in_research_python(self, tmp_path: Path):
        research_py = tmp_path / "research" / "tools" / "x.py"
        research_py.parent.mkdir(parents=True)
        research_py.write_text('P = "/home/someone/x"\n')
        doc = tmp_path / "docs" / "layout.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("tree lives under /home/someone/x\n")
        report = audit([research_py, doc], repo_root=tmp_path)
        assert [v["file"] for v in report["host_path_violations"]] == ["research/tools/x.py"]
        assert [w["file"] for w in report["host_path_warnings"]] == ["docs/layout.md"]
        assert not report["ok"]

    def test_secret_anywhere_fails(self, tmp_path: Path):
        doc = tmp_path / "notes.md"
        doc.write_text(f"the key was {_sandbox_key()}\n")
        report = audit([doc], repo_root=tmp_path)
        assert len(report["secret_findings"]) == 1
        assert not report["ok"]

    def test_clean_tree_ok(self, tmp_path: Path):
        src = tmp_path / "research" / "adapters" / "ok.py"
        src.parent.mkdir(parents=True)
        src.write_text("import os\nkey = os.environ[\"SANDBOX_API_KEY\"]\n")
        report = audit([src], repo_root=tmp_path)
        assert report["ok"]
        assert report["scanned_files"] == 1

    def test_self_referential_files_still_catch_real_keys(self, tmp_path: Path):
        tool = tmp_path / "research" / "tools" / "audit_secrets.py"
        tool.parent.mkdir(parents=True)
        tool.write_text(
            f'REAL = "{_sandbox_key()}"\n'
            f'{_generic_assignment()}\n'
            'HOME = "/home/someone/x"\n'
        )
        report = audit([tool], repo_root=tmp_path)
        # generic assignment and host-path rules are skipped for the audit's
        # own module, but a real key format must still be caught.
        assert [f["pattern"] for f in report["secret_findings"]] == ["openmle_sandbox_key"]
        assert report["host_path_violations"] == []
        assert not report["ok"]
