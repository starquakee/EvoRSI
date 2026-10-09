"""Reproducible path/hash inventory and provenance manifest for US-001.

Walks the three source trees (WSL OpenRSI repo, legacy rsi-gpu tree, Windows
deployed tree) and records every included file as {path, size, sha256}.
Files whose names match secret patterns are recorded as {path, size} ONLY —
their contents are never read, so credentials cannot leak into the output.
Generated/runtime directories (venvs, outputs, experiment data, VCS metadata)
are excluded from hashing and listed by name in the report.

Tree roots are configured in research/tools/inventory_roots.json (data file,
so machine-specific absolute paths stay out of Python source). The inventory's
own output directory is excluded from the openrsi-wsl walk so re-running the
tool does not invalidate its own previous reports.

Usage:
    <venv-python> research/tools/inventory.py                 # print summaries
    <venv-python> research/tools/inventory.py --write         # write JSON reports
    <venv-python> research/tools/inventory.py --manifest      # write provenance manifest
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS_CONFIG = Path(__file__).resolve().parent / "inventory_roots.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "provenance"

# Directory names never descended into (matched anywhere in the tree).
EXCLUDED_DIR_NAMES = frozenset({
    ".git", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache",
    ".idea", ".vscode", "outputs", "sandbox-data", "sandbox-workdir",
    "sandbox-cache", "artifacts", "formal_baseline", ".local-backups",
    ".runtime",
})
EXCLUDED_DIR_GLOBS = (".venv*", "venv", "*.egg-info")

# File names whose contents are never read (likely credential stores).
# Deliberately extension-focused so source files (e.g. audit_secrets.py) are
# still hashed; tracked source is separately scanned by audit_secrets.py.
SECRET_FILE_GLOBS = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx",
    "sandbox-api-key", "id_rsa*", "id_ed25519*",
    "*credentials*", "secrets.*", ".secrets", ".secrets.*",
)
# Public templates that match the secret globs above but contain no secrets
# (mirrors the root .gitignore `!.env.example` carve-out).
SECRET_GLOB_ALLOWLIST = (".env.example", ".env.sample", ".env.template")

_HASH_CHUNK = 1024 * 1024


def _is_secret_name(name: str) -> bool:
    if name in SECRET_GLOB_ALLOWLIST:
        return False
    return any(fnmatch.fnmatch(name, pat) for pat in SECRET_FILE_GLOBS)


def _is_excluded_dir(name: str) -> bool:
    return name in EXCLUDED_DIR_NAMES or any(
        fnmatch.fnmatch(name, pat) for pat in EXCLUDED_DIR_GLOBS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_tree(root: Path, *, skip_dirs: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Walk one tree; never read the contents of secret-named files."""
    root = root.resolve()
    files: list[dict[str, Any]] = []
    excluded_secrets: list[dict[str, Any]] = []
    symlinks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames
            if not _is_excluded_dir(d)
            and not (current / d).is_symlink()
            and (current / d).resolve() not in skip_dirs
        )
        for name in sorted(filenames):
            full = current / name
            rel = full.relative_to(root).as_posix()
            try:
                if full.is_symlink():
                    symlinks.append({"path": rel, "target": os.readlink(full)})
                    continue
                size = full.stat().st_size
                if _is_secret_name(name):
                    excluded_secrets.append({"path": rel, "size": size})
                    continue
                files.append({"path": rel, "size": size, "sha256": _sha256(full)})
            except OSError as exc:
                errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(f["size"] for f in files),
        "excluded_secret_count": len(excluded_secrets),
        "files": files,
        "excluded_secrets": excluded_secrets,
        "symlinks": symlinks,
        "errors": errors,
    }


def diff_inventories(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Surface differences; nothing is ever overwritten by this tool."""
    map_a = {f["path"]: f["sha256"] for f in a["files"]}
    map_b = {f["path"]: f["sha256"] for f in b["files"]}
    common = sorted(set(map_a) & set(map_b))
    return {
        "a": a["root"],
        "b": b["root"],
        "common_count": len(common),
        "identical_count": sum(1 for p in common if map_a[p] == map_b[p]),
        "hash_mismatches": sorted(p for p in common if map_a[p] != map_b[p]),
        "only_in_a": sorted(set(map_a) - set(map_b)),
        "only_in_b": sorted(set(map_b) - set(map_a)),
    }


def load_tree_roots(config_path: Path = DEFAULT_ROOTS_CONFIG) -> list[dict[str, str]]:
    entries = json.loads(config_path.read_text(encoding="utf-8"))["trees"]
    for entry in entries:
        root = Path(entry["root"])
        entry["root"] = str(root if root.is_absolute() else (REPO_ROOT / root).resolve())
    return entries


def _git_field(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _compare_file(origin: Path, dest: Path, note: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "origin_path": str(origin),
        "repo_path": dest.relative_to(REPO_ROOT).as_posix(),
        "origin_sha256": _sha256(origin),
        "repo_sha256": _sha256(dest),
    }
    record["identical"] = record["origin_sha256"] == record["repo_sha256"]
    if note:
        record["note"] = note
    return record


def _compare_dir(origin: Path, dest: Path) -> dict[str, Any]:
    origin_files = {
        p.relative_to(origin).as_posix(): p
        for p in sorted(origin.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
    }
    dest_files = {
        p.relative_to(dest).as_posix(): p
        for p in sorted(dest.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
    }
    common = sorted(set(origin_files) & set(dest_files))
    mismatches = sorted(
        rel for rel in common
        if _sha256(origin_files[rel]) != _sha256(dest_files[rel])
    )
    return {
        "file_count": len(dest_files),
        "mismatches": mismatches,
        "missing_in_repo": sorted(set(origin_files) - set(dest_files)),
        "extra_in_repo": sorted(set(dest_files) - set(origin_files)),
        "identical": not mismatches
        and set(origin_files) == set(dest_files),
    }


def generate_manifest(roots: list[dict[str, str]]) -> dict[str, Any]:
    """Verify consolidated copies against their read-only origins."""
    legacy = Path(roots_by_name(roots, "rsi-gpu-legacy"))
    if not legacy.is_dir():
        raise SystemExit(f"legacy origin tree not available: {legacy}")

    components: list[dict[str, Any]] = []
    file_notes = {
        "rsi_eval_adapter.py": "import rewritten package-relative; gate logic unchanged",
        "safe_rsi_de_adapter.py": "docstring provenance note; __main__ prints real newline",
        "sandbox_eval_client.py": "US-004 unified source gate; US-007 cancellation registry, cleanup proof and bounded polling",
    }
    adapter_files = []
    for name in sorted(p.name for p in (REPO_ROOT / "research/adapters").glob("*.py") if p.name != "__init__.py"):
        adapter_files.append(_compare_file(
            legacy / name, REPO_ROOT / "research/adapters" / name,
            note=file_notes.get(name, "intentional consolidation edit"),
        ))
    components.append({
        "name": "research/adapters",
        "origin_tree": "rsi-gpu-legacy",
        "files": adapter_files,
    })

    experiment_notes = {
        "de_formal_baseline.py": "credentials env-only (key-scraping fallback removed); repo-relative paths",
        "de_search_hello.py": "repo-relative import resolution",
        "smoke_adapter.py": "renamed from e2e/test_adapter.py; repo-relative imports",
    }
    experiment_origin = {
        "de_formal_baseline.py": "e2e/de_formal_baseline.py",
        "de_search_hello.py": "e2e/de_search_hello.py",
        "hello_synth_job.py": "e2e/hello_synth_job.py",
        "submit_e2e.py": "e2e/submit_e2e.py",
        "smoke_adapter.py": "e2e/test_adapter.py",
        "run_istratde_gpu.py": "run_istratde_gpu.py",
        "run_metade_gpu.py": "run_metade_gpu.py",
    }
    experiment_files = []
    for dest_name, origin_rel in sorted(experiment_origin.items()):
        experiment_files.append(_compare_file(
            legacy / origin_rel, REPO_ROOT / "research/experiments" / dest_name,
            note=experiment_notes.get(dest_name),
        ))
    components.append({
        "name": "research/experiments",
        "origin_tree": "rsi-gpu-legacy",
        "files": experiment_files,
    })

    for pkg in ("istratde", "metade"):
        origin_repo = legacy / pkg
        vcs = {
            "url": _git_field(origin_repo, "remote", "get-url", "origin"),
            "commit": _git_field(origin_repo, "rev-parse", "HEAD"),
            "dirty": bool(_git_field(origin_repo, "status", "--porcelain")),
        }
        components.append({
            "name": f"research/vendor/{pkg}",
            "origin_tree": "rsi-gpu-legacy",
            "origin_path": str(origin_repo),
            "license": "GPL-3.0 (LICENSE preserved)",
            "vcs": vcs,
            "copied": ["src/", "pyproject.toml", "LICENSE", "README.md"],
            "comparison": _compare_dir(origin_repo / "src", REPO_ROOT / "research/vendor" / pkg / "src"),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "research/tools/inventory.py --manifest",
        "repo_root": str(REPO_ROOT),
        "components": components,
        "environment_snapshots": [
            "research/provenance/venv-rsi-gpu.freeze.txt",
            "research/provenance/venv-openmle-evo.freeze.txt",
            "research/provenance/venv-research.freeze.txt",
        ],
    }


def roots_by_name(roots: list[dict[str, str]], name: str) -> str:
    for entry in roots:
        if entry["name"] == name:
            return entry["root"]
    raise KeyError(f"tree {name!r} not configured")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, default=DEFAULT_ROOTS_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write", action="store_true", help="write per-tree JSON reports")
    parser.add_argument("--manifest", action="store_true", help="write provenance manifest")
    args = parser.parse_args(argv)

    roots = load_tree_roots(args.roots)
    skip_dirs = frozenset({args.output.resolve()})
    inventories: dict[str, dict[str, Any]] = {}
    for entry in roots:
        root = Path(entry["root"])
        if not root.is_dir():
            print(f"SKIP {entry['name']}: {root} not available", flush=True)
            continue
        inv = inventory_tree(root, skip_dirs=skip_dirs)
        inv["tree"] = entry["name"]
        inventories[entry["name"]] = inv
        print(
            f"{entry['name']}: {inv['file_count']} files, "
            f"{inv['excluded_secret_count']} secret-named (not read), "
            f"{len(inv['errors'])} errors",
            flush=True,
        )
        if args.write:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f"inventory-{entry['name']}.json").write_text(
                json.dumps(inv, indent=2) + "\n", encoding="utf-8"
            )

    pair = ("openrsi-wsl", "windows-deployed")
    if all(name in inventories for name in pair):
        diff = diff_inventories(inventories[pair[0]], inventories[pair[1]])
        print(
            f"diff {pair[0]} vs {pair[1]}: {diff['identical_count']}/{diff['common_count']} identical, "
            f"{len(diff['hash_mismatches'])} mismatched, "
            f"{len(diff['only_in_a'])} only-in-wsl, {len(diff['only_in_b'])} only-in-windows",
            flush=True,
        )
        if args.write:
            (args.output / "inventory-diff-wsl-vs-windows.json").write_text(
                json.dumps(diff, indent=2) + "\n", encoding="utf-8"
            )

    if args.manifest:
        manifest = generate_manifest(roots)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        bad = [
            f["repo_path"]
            for comp in manifest["components"] if "files" in comp
            for f in comp["files"] if not f["identical"] and "note" not in f
        ]
        bad += [
            comp["name"]
            for comp in manifest["components"] if "comparison" in comp
            and not comp["comparison"]["identical"]
        ]
        if bad:
            print(f"MANIFEST UNEXPLAINED DIFFERENCES: {bad}", flush=True)
            return 1
        print("manifest: all copies verified (differences enumerated with notes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
