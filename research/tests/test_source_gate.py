"""Contract tests for the unified pre-execution source gate (US-004).

The gate checks immutable submitted bytes against a versioned policy
artifact and returns machine-readable denials. It is an advisory admission
filter and must never be represented as a security boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.contracts import source_gate
from research.contracts.source_gate import (
    DEFAULT_POLICY_PATH,
    GateDenial,
    SourcePolicyError,
    SourceVerdict,
    check_source_bytes,
    check_source_tree,
    load_policy,
    policy_unavailable_verdict,
    validate_job_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENIGN_SOURCE = (REPO_ROOT / "research/experiments/hello_synth_job.py").read_bytes()


@pytest.fixture()
def policy():
    return load_policy()


class TestPolicyLoading:
    def test_bundled_policy_loads_and_is_versioned(self, policy):
        assert policy.policy_version == "source-gate.v2"
        assert policy.policy_sha256 == hashlib.sha256(
            DEFAULT_POLICY_PATH.read_bytes()
        ).hexdigest()
        assert policy.security_boundary is False

    def test_missing_policy_fails_closed(self, tmp_path):
        with pytest.raises(SourcePolicyError, match="policy_unavailable"):
            load_policy(tmp_path / "does-not-exist.json")

    def test_corrupt_policy_fails_closed(self, tmp_path):
        bad = tmp_path / "policy.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(SourcePolicyError, match="policy_corrupt"):
            load_policy(bad)

    def test_wrong_schema_version_fails_closed(self, tmp_path):
        bad = tmp_path / "policy.json"
        bad.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
        with pytest.raises(SourcePolicyError, match="policy_schema_version_unsupported"):
            load_policy(bad)

    @pytest.mark.parametrize(
        "mutation",
        [
            {"policy_version": ""},
            {"max_source_bytes": 0},
            {"denied_source_markers": "not-a-list"},
            {"allow_execution_command": "yes"},
            {"requirement_pattern": "([invalid"},
        ],
    )
    def test_invalid_fields_fail_closed(self, tmp_path, mutation):
        raw = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
        raw.update(mutation)
        bad = tmp_path / "policy.json"
        bad.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SourcePolicyError):
            load_policy(bad)

    def test_policy_unavailable_verdict_denies(self):
        verdict = policy_unavailable_verdict("policy_unavailable:FileNotFoundError", b"x = 1\n")
        assert not verdict.allowed
        assert verdict.to_dict()["denials"][0]["rule"] == "policy_unavailable"
        assert verdict.source_sha256 == hashlib.sha256(b"x = 1\n").hexdigest()


class TestSourceBytes:
    def test_benign_ml_code_allowed(self, policy):
        verdict = check_source_bytes(BENIGN_SOURCE, policy)
        assert verdict.allowed, verdict.to_dict()
        assert verdict.denials == ()

    def test_immutable_bytes_hashed_exactly(self, policy):
        source = "import pandas as pd\nprint(pd.__version__)\n".encode()
        verdict = check_source_bytes(source, policy)
        assert verdict.source_sha256 == hashlib.sha256(source).hexdigest()
        # Trailing byte changes the bound hash.
        other = check_source_bytes(source + b"\n", policy)
        assert other.source_sha256 != verdict.source_sha256

    def test_verdict_binds_policy_version_and_hash(self, policy):
        verdict = check_source_bytes(BENIGN_SOURCE, policy)
        assert verdict.policy_version == policy.policy_version
        assert verdict.policy_sha256 == policy.policy_sha256

    def test_verdict_is_machine_readable_and_not_a_security_boundary(self, policy):
        doc = check_source_bytes(BENIGN_SOURCE, policy).to_dict()
        assert doc["security_boundary"] is False
        assert doc["check_kind"] == "static_admission_filter"
        assert set(doc) >= {
            "allowed",
            "policy_version",
            "policy_sha256",
            "source_sha256",
            "denials",
        }

    @pytest.mark.parametrize(
        "marker",
        [
            "subprocess",
            "os.system",
            "docker.sock",
            "read_and_metric",
            "/evaluation/",
            "urllib.request",
            "ctypes",
            "../",
        ],
    )
    def test_malicious_markers_denied_with_machine_readable_rule(self, policy, marker):
        source = f"payload = {marker!r}\n{marker}_call()\n".encode()
        verdict = check_source_bytes(source, policy)
        assert not verdict.allowed
        rules = [d.rule for d in verdict.denials]
        assert "denied_marker" in rules
        denial = verdict.to_dict()["denials"][0]
        assert set(denial) == {"rule", "reason"}

    def test_marker_matching_is_case_insensitive(self, policy):
        verdict = check_source_bytes(b"import OS\nOS.SYSTEM('id')\n", policy)
        assert not verdict.allowed

    def test_empty_source_denied(self, policy):
        verdict = check_source_bytes(b"   \n", policy)
        assert not verdict.allowed
        assert verdict.denials[0].rule == "empty_source"

    def test_oversized_source_denied(self, policy):
        verdict = check_source_bytes(b"a" * (policy.max_source_bytes + 1), policy)
        assert not verdict.allowed
        assert any(d.rule == "source_too_large" for d in verdict.denials)

    def test_non_utf8_source_denied(self, policy):
        verdict = check_source_bytes(b"\xff\xfe\x00bad", policy)
        assert not verdict.allowed
        assert any(d.rule == "source_not_utf8" for d in verdict.denials)


class TestSourceTree:
    def test_single_file_matches_bytes_semantics(self, policy, tmp_path):
        target = tmp_path / "main.py"
        target.write_bytes(BENIGN_SOURCE)
        verdict = check_source_tree(target, policy)
        assert verdict.allowed, verdict.to_dict()

    def test_directory_checks_every_python_file(self, policy, tmp_path):
        (tmp_path / "good.py").write_text("import pandas as pd\n", encoding="utf-8")
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "evil.py").write_text("import subprocess\n", encoding="utf-8")
        (sub / "data.txt").write_text("subprocess in data is not code\n", encoding="utf-8")
        verdict = check_source_tree(tmp_path, policy)
        assert not verdict.allowed
        assert any(d.rule == "denied_marker" for d in verdict.denials)

    def test_tree_hash_changes_when_file_swapped(self, policy, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        first = check_source_tree(tmp_path, policy)
        (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
        second = check_source_tree(tmp_path, policy)
        assert first.allowed and second.allowed
        assert first.source_sha256 != second.source_sha256

    def test_directory_without_python_denied(self, policy, tmp_path):
        (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
        verdict = check_source_tree(tmp_path, policy)
        assert not verdict.allowed
        assert any(d.rule == "empty_source" for d in verdict.denials)

    def test_symlink_denied(self, policy, tmp_path):
        real = tmp_path / "real.py"
        real.write_text("x = 1\n", encoding="utf-8")
        link = tmp_path / "link.py"
        link.symlink_to(real)
        real.unlink()
        verdict = check_source_tree(link, policy)
        assert not verdict.allowed
        assert any(d.rule == "symlink_denied" for d in verdict.denials)

    def test_missing_artifact_fails_closed(self, policy, tmp_path):
        verdict = check_source_tree(tmp_path / "nope.py", policy)
        assert not verdict.allowed
        assert any(d.rule == "artifact_missing" for d in verdict.denials)


class TestJobFieldValidation:
    def test_legitimate_fields_pass(self, policy):
        denials = validate_job_fields(
            policy,
            execution_command=None,
            requirements=["lightgbm==4.5.0", "scikit-learn"],
            environment={
                "EXECUTION_MODE": "shell",
                "DATA_DIR": "/data/tasks/hello_synth",
                "SANDBOX_DATA_DIR": "/data/tasks/hello_synth",
                "EVAL_SPLIT": "validation",
            },
            working_dir=None,
            code_file_path=None,
            data_paths=["/data/tasks/hello_synth/train.csv"],
        )
        assert denials == ()

    def test_execution_command_denied_machine_readable(self, policy):
        denials = validate_job_fields(policy, execution_command="bash run.sh")
        assert [d.rule for d in denials] == ["execution_command_denied"]

    @pytest.mark.parametrize(
        "entry",
        [
            "git+https://example.invalid/repo.git",
            "https://example.invalid/pkg.whl",
            "-r requirements.txt",
            "./local/path",
            "pkg @ https://example.invalid/pkg.whl",
        ],
    )
    def test_unsafe_requirements_denied(self, policy, entry):
        denials = validate_job_fields(policy, requirements=[entry])
        assert any(d.rule == "requirement_not_allowlisted" for d in denials)

    @pytest.mark.parametrize(
        "key",
        [
            "LD_PRELOAD",
            "ld_library_path",
            "PYTHONPATH",
            "BASH_ENV",
            "PYTHONSTARTUP",
            "EXECUTION_COMMAND",
            "JOB_OUTPUT_DIR",
            "SOURCE_GATE_POLICY_PATH",
            "EVALUATOR_ANSWER_PATH",
            "BUDGET_LEDGER_PATH",
            # v2: loader/platform bypasses and worker-control variables
            "PATH",
            "HOME",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "CDPATH",
            "SANDBOX_API_KEYS",
            "WORKER_CONTROL_TOKEN",
            "REDIS_HOST",
            "DB_PASSWORD",
            "POSTGRES_PASSWORD",
            "DOCKER_HOST",
        ],
    )
    def test_dangerous_environment_keys_denied(self, policy, key):
        denials = validate_job_fields(policy, environment={key: "x"})
        assert any(d.rule == "environment_key_denied" for d in denials), key

    def test_working_dir_traversal_denied(self, policy):
        denials = validate_job_fields(policy, working_dir="/data/../secrets")
        rules = [d.rule for d in denials]
        assert "path_traversal" in rules
        assert "path_protected_component" in rules

    def test_data_dir_traversal_denied(self, policy):
        denials = validate_job_fields(policy, data_dir="/data/../evaluation")
        rules = [d.rule for d in denials]
        assert "path_traversal" in rules
        assert "path_protected_component" in rules

    def test_data_dir_protected_component_denied(self, policy):
        denials = validate_job_fields(policy, data_dir="/data/tasks/answers")
        assert any(d.rule == "path_protected_component" for d in denials)

    def test_legitimate_data_dir_passes(self, policy):
        denials = validate_job_fields(
            policy,
            data_dir="/mnt/pubdatasets2/tasks/hello_synth",
            environment={"DATA_DIR": "/mnt/pubdatasets2/tasks/hello_synth"},
        )
        assert denials == ()

    @pytest.mark.parametrize("key", ["DATA_DIR", "SANDBOX_DATA_DIR"])
    def test_path_valued_environment_values_checked(self, policy, key):
        denials = validate_job_fields(policy, environment={key: "/data/../answers"})
        rules = [d.rule for d in denials]
        assert "path_traversal" in rules
        assert "path_protected_component" in rules

    @pytest.mark.parametrize(
        "field, value",
        [
            ("code_file_path", "/data/evaluation/read_and_metric.py"),
            ("working_dir", "/data/tasks/answer"),
            ("data_paths", ["/data/answers/private.csv"]),
        ],
    )
    def test_protected_path_components_denied(self, policy, field, value):
        kwargs = {field: value}
        denials = validate_job_fields(policy, **kwargs)
        assert any(d.rule == "path_protected_component" for d in denials)


def test_gate_module_exported_from_contracts():
    from research.contracts import check_source_bytes as exported  # noqa: F401

    assert exported is source_gate.check_source_bytes


def test_verdict_frozen_immutable(policy):
    verdict = check_source_bytes(b"import subprocess\n", policy)
    assert isinstance(verdict, SourceVerdict)
    assert verdict.denials
    with pytest.raises(AttributeError):
        verdict.allowed = True
    with pytest.raises(AttributeError):
        verdict.denials[0].rule = "tampered"
    assert isinstance(verdict.denials[0], GateDenial)
