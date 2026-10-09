"""Contract tests for research/tools/inventory.py (US-001).

Pins the security-critical behavior: secret-named files are never read (no
hash, no content), generated/runtime directories are excluded, output is
deterministic, and diffs surface mismatches instead of overwriting anything.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.tools.inventory import (
    diff_inventories,
    inventory_tree,
    load_tree_roots,
    roots_by_name,
)


@pytest.fixture()
def sample_tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("print('a')\n")
    (root / "README.md").write_text("hello\n")
    (root / ".env").write_text("SHOULD_NEVER_BE_READ=1\n")
    (root / ".env.example").write_text("TEMPLATE_VAR=\n")
    (root / "credentials.json").write_text("{}\n")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "x.py").write_text("x = 1\n")
    (root / "formal_baseline").mkdir()
    (root / "formal_baseline" / "eval_log.jsonl").write_text("{}\n")
    (root / "sandbox-data").mkdir()
    (root / "sandbox-data" / "blob").write_bytes(b"\x00" * 8)
    return root


class TestInventoryTree:
    def test_hashes_regular_files(self, sample_tree: Path):
        inv = inventory_tree(sample_tree)
        by_path = {f["path"]: f for f in inv["files"]}
        expected = hashlib.sha256(b"print('a')\n").hexdigest()
        assert by_path["src/a.py"]["sha256"] == expected
        assert "README.md" in by_path

    def test_secret_files_recorded_without_hash(self, sample_tree: Path):
        inv = inventory_tree(sample_tree)
        secret_paths = {s["path"] for s in inv["excluded_secrets"]}
        assert secret_paths == {".env", "credentials.json"}
        for entry in inv["excluded_secrets"]:
            assert "sha256" not in entry
        hashed = {f["path"] for f in inv["files"]}
        assert ".env" not in hashed
        assert "credentials.json" not in hashed

    def test_env_template_is_hashed(self, sample_tree: Path):
        inv = inventory_tree(sample_tree)
        hashed = {f["path"] for f in inv["files"]}
        assert ".env.example" in hashed

    def test_generated_dirs_excluded(self, sample_tree: Path):
        inv = inventory_tree(sample_tree)
        all_paths = {f["path"] for f in inv["files"]}
        assert not any(p.startswith((".venv/", "formal_baseline/", "sandbox-data/")) for p in all_paths)

    def test_deterministic_file_order(self, sample_tree: Path):
        first = [f["path"] for f in inventory_tree(sample_tree)["files"]]
        second = [f["path"] for f in inventory_tree(sample_tree)["files"]]
        assert first == second == sorted(first)

    def test_symlinks_not_followed(self, sample_tree: Path):
        outside = sample_tree.parent / "outside.txt"
        outside.write_text("secret-ish\n")
        (sample_tree / "link.txt").symlink_to(outside)
        inv = inventory_tree(sample_tree)
        assert [s["path"] for s in inv["symlinks"]] == ["link.txt"]
        assert "link.txt" not in {f["path"] for f in inv["files"]}


class TestDiff:
    @staticmethod
    def _inv(root: str, entries: dict[str, str]) -> dict:
        return {
            "root": root,
            "files": [
                {"path": path, "size": len(content), "sha256": hashlib.sha256(content.encode()).hexdigest()}
                for path, content in entries.items()
            ],
        }

    def test_diff_surfaces_all_categories(self):
        a = self._inv("a", {"same.py": "x", "diff.py": "old", "only_a.py": "1"})
        b = self._inv("b", {"same.py": "x", "diff.py": "new", "only_b.py": "2"})
        diff = diff_inventories(a, b)
        assert diff["common_count"] == 2
        assert diff["identical_count"] == 1
        assert diff["hash_mismatches"] == ["diff.py"]
        assert diff["only_in_a"] == ["only_a.py"]
        assert diff["only_in_b"] == ["only_b.py"]


class TestRootsConfig:
    def test_repo_tree_resolves_relative(self):
        roots = load_tree_roots()
        names = {entry["name"] for entry in roots}
        assert names == {"openrsi-wsl", "rsi-gpu-legacy", "windows-deployed"}
        from research.tools.inventory import REPO_ROOT
        assert roots_by_name(roots, "openrsi-wsl") == str(REPO_ROOT)
