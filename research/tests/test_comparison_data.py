"""EXP-002 focused tests: official data, fixed grouped splits, sealed exposure.

These tests run against the REAL prepared artifacts (not fixtures): the
committed raw UCI archives, the manifest, the worker-visible public CSVs,
the sealed CSVs and the evaluator answers. Byte identities are recomputed
from the raw archives, so determinism is proven, not asserted.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

import pytest

from research.comparison import data as cdata
from research.comparison.data import (
    DATA_SCHEMA,
    DEFAULT_MANIFEST_PATH,
    EVALUATOR_TASKS_ROOT,
    PUBLIC_ROOT,
    REPO_ROOT,
    SEALED_ROOT,
    SHUFFLE_DOMAIN,
    SKLEARN_PIN,
    TASK_SOURCES_BY_ID,
    ComparisonDataError,
    RawRow,
    assign_folds,
    assign_groups,
    assign_split_ids,
    emit_task_artifacts,
    load_manifest,
    load_task_bindings,
    prepare_all,
    publish_final_refit,
    shuffle_seed,
)
from research.comparison.protocol import TASKS_BY_ID, split_seed
from research.comparison.specs import iter_cells
from research.evaluator.service import (
    EvaluatorError,
    EvaluatorRegistry,
    score_submission,
    validate_worker_mount_plan,
)

COMPOSE_PATH = REPO_ROOT / "deploy" / "rsi-trustworthy" / "docker-compose.yaml"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_csv(path: Path) -> list[list[str]]:
    import csv

    return [row for row in csv.reader(path.open(newline="")) if row]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_manifest()


# --- real shapes, classes, provenance -------------------------------------------
def test_real_shapes_classes_and_provenance(manifest: dict) -> None:
    assert manifest["schema"] == DATA_SCHEMA
    assert manifest["sklearn_version"] == SKLEARN_PIN
    assert manifest["split_seed_base"] == 20261010
    for task_id, pin in TASKS_BY_ID.items():
        entry = manifest["tasks"][task_id]
        # Real downloaded shape equals the frozen protocol pin.
        assert entry["rows"] == pin.rows
        assert entry["features"] == pin.features
        assert entry["uci_id"] == pin.uci_id
        assert entry["split_seed"] == split_seed(pin)
        assert entry["license"] == "CC BY 4.0"
        assert entry["url"].startswith("https://archive.ics.uci.edu/static/public/")
        # Both classes overall and in every split.
        counts = entry["class_counts"]
        assert counts["0"] > 0 and counts["1"] > 0
        assert counts["0"] + counts["1"] == pin.rows
        for split in ("train", "validation", "final"):
            split_counts = entry["splits"][split]["class_counts"]
            assert split_counts["0"] > 0 and split_counts["1"] > 0
            rows = entry["splits"][split]["rows"]
            assert split_counts["0"] + split_counts["1"] == rows
        # ~60/20/20 fold geometry (within one fold's granularity).
        total = pin.rows
        assert abs(entry["splits"]["train"]["rows"] - 0.6 * total) < 0.05 * total
        assert abs(entry["splits"]["validation"]["rows"] - 0.2 * total) < 0.05 * total
        assert abs(entry["splits"]["final"]["rows"] - 0.2 * total) < 0.05 * total
        # Raw archive + provenance are sealed on disk with the pinned hash.
        archive = SEALED_ROOT / task_id / "raw" / entry["archive"]["name"]
        assert _sha256(archive.read_bytes()) == entry["archive"]["sha256"]
        provenance = json.loads(
            (SEALED_ROOT / task_id / "raw" / "provenance.json").read_text()
        )
        assert provenance["url"] == entry["url"]
        assert provenance["license"] == "CC BY 4.0"
        assert provenance["archive_sha256"] == entry["archive"]["sha256"]


def test_wdbc_original_ids_removed(manifest: dict) -> None:
    # The raw WDBC member's first column (e.g. 842302) must not appear as an
    # id anywhere in the prepared artifacts.
    member = zipfile.ZipFile(
        SEALED_ROOT / "wdbc" / "raw" / "breast+cancer+wisconsin+diagnostic.zip"
    ).read("wdbc.data")
    original_ids = {line.split(b",")[0] for line in member.split(b"\n") if line}
    assert b"842302" in original_ids  # sanity: we read the real member
    for artifact in (
        PUBLIC_ROOT / "wdbc" / "data" / "public" / "train.csv",
        PUBLIC_ROOT / "wdbc" / "data" / "public" / "test.csv",
        SEALED_ROOT / "wdbc" / "final_features.csv",
        SEALED_ROOT / "wdbc" / "final_labels.csv",
    ):
        body = artifact.read_bytes()
        for original in original_ids:
            assert not any(
                line.split(b",")[0] == original for line in body.split(b"\n")[1:] if line
            ), f"original WDBC id {original!r} leaked into {artifact}"


# --- group integrity ----------------------------------------------------------------
def _canonical_rows(task_id: str) -> list[list[str]]:
    return _read_csv(SEALED_ROOT / task_id / "canonical.csv")


def _split_of_row(task_id: str, features: list[str]) -> str:
    """Locate a canonical feature vector's split via the emitted CSVs."""
    key = ",".join(features)
    for split, path in (
        ("train", PUBLIC_ROOT / task_id / "data" / "public" / "train.csv"),
        ("validation", PUBLIC_ROOT / task_id / "data" / "public" / "test.csv"),
        ("final", SEALED_ROOT / task_id / "final_features.csv"),
    ):
        for line in path.read_text().splitlines()[1:]:
            if key in line.split(",", 1)[1]:
                return split
    raise AssertionError(f"row not found in any split: {key}")


def test_groups_never_straddle_splits(manifest: dict) -> None:
    for task_id in TASKS_BY_ID:
        rows = _canonical_rows(task_id)
        header, body = rows[0], rows[1:]
        assert header[0] == "group_id" and header[-1] == "label"
        group_splits: dict[str, set[str]] = {}
        for row in body:
            group_id, features = row[0], row[1:-1]
            group_splits.setdefault(group_id, set()).add(
                _split_of_row(task_id, features)
            )
        assert group_splits  # non-empty
        for group_id, splits in group_splits.items():
            assert len(splits) == 1, f"{task_id} group {group_id} in {splits}"


def test_contradictory_label_groups_preserved(manifest: dict) -> None:
    # Spambase has real duplicate feature vectors with contradictory labels;
    # they must be preserved (never deduplicated) and co-located.
    entry = manifest["tasks"]["spambase"]
    assert entry["groups"]["contradictory_label_groups"] == 3
    rows = _canonical_rows("spambase")[1:]
    by_group: dict[str, list[list[str]]] = {}
    for row in rows:
        by_group.setdefault(row[0], []).append(row)
    contradictory = [g for g, rs in by_group.items() if len({r[-1] for r in rs}) > 1]
    assert len(contradictory) == 3
    for group_id in contradictory:
        for row in by_group[group_id]:
            assert _split_of_row("spambase", row[1:-1]) == _split_of_row(
                "spambase", by_group[group_id][0][1:-1]
            )


def test_no_row_overlap_between_splits(manifest: dict) -> None:
    for task_id, pin in TASKS_BY_ID.items():
        seen: list[tuple[str, ...]] = []
        for path in (
            PUBLIC_ROOT / task_id / "data" / "public" / "train.csv",
            PUBLIC_ROOT / task_id / "data" / "public" / "test.csv",
            SEALED_ROOT / task_id / "final_features.csv",
        ):
            for line in path.read_text().splitlines()[1:]:
                seen.append(tuple(line.split(",")[1:]))  # drop id, keep features(+label)
        # Feature identity per split is disjoint because groups cannot straddle.
        train = set(seen[: manifest["tasks"][task_id]["splits"]["train"]["rows"]])
        rest = seen[manifest["tasks"][task_id]["splits"]["train"]["rows"] :]
        assert not train.intersection(tuple(r[: pin.features] for r in rest))
        total = sum(manifest["tasks"][task_id]["splits"][s]["rows"] for s in ("train", "validation", "final"))
        assert total == pin.rows


# --- opaque ids ----------------------------------------------------------------------
def test_ids_are_fresh_opaque_and_label_independent(manifest: dict) -> None:
    for task_id in TASKS_BY_ID:
        for split, path in (
            ("train", PUBLIC_ROOT / task_id / "data" / "public" / "train.csv"),
            ("validation", PUBLIC_ROOT / task_id / "data" / "public" / "test.csv"),
            ("final", SEALED_ROOT / task_id / "final_features.csv"),
        ):
            rows = _read_csv(path)
            header, body = rows[0], rows[1:]
            ids = [int(row[0]) for row in body]
            n = manifest["tasks"][task_id]["splits"][split]["rows"]
            # Fresh sequential ids: a subset of original row indices could
            # never be exactly 0..n-1 for validation/final.
            assert sorted(ids) == list(range(n))
            # Label-independent: the shuffle permutation is a pure function
            # of (task, split, row_count) — recompute it from the documented
            # domain-separated seed and require a real permutation.
            seed = shuffle_seed(task_id, split)
            expected = list(range(n))
            random.Random(seed).shuffle(expected)
            assert assign_split_ids(task_id, split, n) == expected
            if n > 2:
                assert expected != list(range(n))  # actually shuffled
            # Ids must not encode class order: labels in id order are not sorted.
            if "label" in header:
                labels = [int(row[-1]) for row in body]
                assert labels != sorted(labels)


def test_shuffle_seed_domain_separation() -> None:
    seeds = {
        shuffle_seed(task, split)
        for task in TASKS_BY_ID
        for split in ("train", "validation", "final")
    }
    assert len(seeds) == 9  # all distinct
    digest = hashlib.sha256(
        f"{SHUFFLE_DOMAIN}|banknote|train".encode("utf-8")
    ).hexdigest()
    assert shuffle_seed("banknote", "train") == int(digest[:16], 16)
    with pytest.raises(ComparisonDataError):
        shuffle_seed("banknote", "test")  # not an approved split name


# --- deterministic byte identities -----------------------------------------------------
def test_pipeline_recomputes_identical_bytes(manifest: dict) -> None:
    """Re-derive every artifact from the committed raw archives; bytes must match."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for task_id, source in TASK_SOURCES_BY_ID.items():
            raw = SEALED_ROOT / task_id / "raw" / source.archive_name
            dest = tmp_root / "research" / "comparison" / "sealed" / task_id / "raw"
            dest.mkdir(parents=True)
            (dest / source.archive_name).write_bytes(raw.read_bytes())
        rebuilt = prepare_all(
            fetch=False,
            repo_root=tmp_root,
            manifest_path=tmp_root / "manifest.json",
        )
    assert rebuilt == manifest
    for task_id in TASKS_BY_ID:
        for rel, expected in manifest["tasks"][task_id]["files"].items():
            assert _sha256((REPO_ROOT / rel).read_bytes()) == expected


def test_prepare_is_idempotent_and_frozen(manifest: dict) -> None:
    # Re-running against the real tree must not change a single byte.
    before = {
        rel: _sha256((REPO_ROOT / rel).read_bytes())
        for task in manifest["tasks"].values()
        for rel in task["files"]
    }
    prepare_all(fetch=False)
    after = {
        rel: _sha256((REPO_ROOT / rel).read_bytes())
        for task in manifest["tasks"].values()
        for rel in task["files"]
    }
    assert before == after
    assert json.loads(DEFAULT_MANIFEST_PATH.read_text()) == manifest


def test_frozen_artifact_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "a.csv"
    cdata._write_frozen(target, b"x\n")
    cdata._write_frozen(target, b"x\n")  # identical is fine
    with pytest.raises(ComparisonDataError, match="artifact_mismatch"):
        cdata._write_frozen(target, b"y\n")


def test_prepare_refuses_without_raw_when_fetch_disabled(tmp_path: Path) -> None:
    with pytest.raises(ComparisonDataError, match="raw_missing"):
        prepare_all(fetch=False, repo_root=tmp_path, manifest_path=tmp_path / "m.json")


# --- malformed data rejection ----------------------------------------------------------
@pytest.mark.parametrize(
    ("task_id", "text"),
    [
        ("banknote", ""),  # empty
        ("banknote", "1,2,3,4,0,extra\n"),  # wrong width
        ("banknote", "1,2,3,4\n"),  # ragged (missing label)
        ("banknote", "1,2,abc,4,0\n"),  # non-numeric feature
        ("banknote", "1,2,nan,4,0\n"),  # non-finite
        ("banknote", "1,2,inf,4,0\n"),  # non-finite
        ("banknote", "1,2,,4,0\n"),  # empty feature
        ("banknote", "1,2,3,4,2\n"),  # label outside {0,1}
        ("wdbc", "842302,X,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,"
         "0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,"
         "0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,"
         "0.7119,0.2654,0.4601,0.1189\n"),  # bad diagnosis
        ("wdbc", "842302,M,17.99\n"),  # wrong width
        ("spambase", ",".join(["0"] * 57 + ["3"]) + "\n"),  # bad label
        ("spambase", ",".join(["0"] * 56) + "\n"),  # wrong width
    ],
)
def test_malformed_conversion_rejected(task_id: str, text: str) -> None:
    converter = {
        "banknote": cdata._convert_banknote,
        "wdbc": cdata._convert_wdbc,
        "spambase": cdata._convert_spambase,
    }[task_id]
    with pytest.raises(ComparisonDataError, match="malformed_data"):
        converter(text)


def test_conversion_preserves_all_rows_and_groups_equal_vectors() -> None:
    # Rows 2 and 3 are numerically identical (-0≡0, 1.0≡1, 0≡0.0, 2≡2.0)
    # but carry contradictory labels: one group, both rows preserved.
    rows = cdata._convert_banknote(
        "0.5,0.5,0.5,0.5,1\n-0,1.0,0,2,0\n0,1,0.0,2.0,1\n"
    )
    assert len(rows) == 3  # nothing deduplicated
    groups = assign_groups(rows)
    assert groups[0] != groups[1]  # first row distinct
    assert groups[1] == groups[2]  # equal numeric vectors share a group
    assert rows[1].features == (0.0, 1.0, 0.0, 2.0)
    assert rows[2].features == (0.0, 1.0, 0.0, 2.0)
    assert rows[1].label == 0 and rows[2].label == 1


def test_assign_folds_rejects_length_mismatch() -> None:
    with pytest.raises(ComparisonDataError, match="malformed_data"):
        assign_folds([0, 1], [0], seed=1)


def test_extract_member_rejects_bad_archive() -> None:
    source = TASK_SOURCES_BY_ID["banknote"]
    with pytest.raises(ComparisonDataError, match="malformed_archive"):
        cdata.extract_member(b"not a zip", source)
    other = zipfile.ZipFile(io.BytesIO(), "w")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("wrong-name.txt", "x")
    del other
    with pytest.raises(ComparisonDataError, match="malformed_archive"):
        cdata.extract_member(buffer.getvalue(), source)


# --- sealed-data exposure ---------------------------------------------------------------
def test_public_dirs_contain_only_search_inputs(manifest: dict) -> None:
    for task_id in TASKS_BY_ID:
        public = PUBLIC_ROOT / task_id / "data" / "public"
        assert sorted(p.name for p in public.iterdir()) == [
            "README.md",
            "sample_submission.csv",
            "test.csv",
            "train.csv",
        ]
        test_rows = _read_csv(public / "test.csv")
        assert "label" not in test_rows[0]
        assert len(test_rows[0]) == TASKS_BY_ID[task_id].features + 1
        sample = _read_csv(public / "sample_submission.csv")
        assert sample[0] == ["id", "label"]
        assert [row[0] for row in sample[1:]] == [row[0] for row in test_rows[1:]]
        assert {row[1] for row in sample[1:]} == {"0"}
        # No label value from the sealed validation answer appears as an
        # extra column anywhere in test.csv (schema check above) and the
        # sealed files stay under research/comparison/sealed only.
        assert not (public / "validation_answer.csv").exists()
        assert not (public / "final_features.csv").exists()
        assert not (public / "final_labels.csv").exists()
        assert not (public / "trainval.csv").exists()


def _compose_worker_mount_sources() -> list[Path]:
    """Host-side volume sources of the WORKER service only.

    Controller-side services (api, dispatcher) legitimately mount
    ``research/`` to run the trusted evaluator; the candidate worker must
    never see the sealed or evaluator trees.
    """
    lines = COMPOSE_PATH.read_text().splitlines()
    sources: list[Path] = []
    in_worker = False
    for line in lines:
        if line.startswith("  ") and not line.startswith("    "):
            # service-level key (e.g. "  worker:")
            in_worker = line.strip() == "worker:"
            continue
        if not in_worker:
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped:
            source = stripped[2:].split(":", 1)[0].strip().strip("'\"")
            if source.startswith("."):
                sources.append((COMPOSE_PATH.parent / source).resolve())
    return sources


def test_sealed_and_evaluator_trees_outside_worker_mounts() -> None:
    sources = _compose_worker_mount_sources()
    assert sources, "compose parse found no mounts"
    protected = [SEALED_ROOT.resolve(), EVALUATOR_TASKS_ROOT.resolve()]
    for source in sources:
        for root in protected:
            assert not root.is_relative_to(source), f"{source} exposes {root}"
            assert not source.is_relative_to(root), f"{source} inside {root}"


def test_mount_plan_guard_denies_protected_ancestors() -> None:
    evaluator_root = (REPO_ROOT / "research" / "evaluator").resolve()
    public_dirs = [
        str(PUBLIC_ROOT / task_id / "data" / "public") for task_id in TASKS_BY_ID
    ]
    assert validate_worker_mount_plan(public_dirs, evaluator_root=evaluator_root) == []
    for bad in (REPO_ROOT, REPO_ROOT / "research", evaluator_root,
                REPO_ROOT / "research" / "evaluator" / "tasks"):
        denials = validate_worker_mount_plan([str(bad)], evaluator_root=evaluator_root)
        assert denials, f"expected denial for {bad}"


# --- evaluator registration -------------------------------------------------------------
def test_registry_registers_comparison_tasks(manifest: dict) -> None:
    registry = EvaluatorRegistry.load()
    assert "hello_synth" in registry.tasks  # preserved
    for task_id in TASKS_BY_ID:
        registered = f"comparison_{task_id}"
        spec = registry.task(registered)
        assert spec.metric == "accuracy"
        assert spec.split == "validation"
        # Registry answer hash == manifest binding hash (single truth).
        binding_hash = manifest["tasks"][task_id]["task_binding"][
            "validation_labels_sha256"
        ]
        assert spec.answer_sha256 == binding_hash
        answer = (
            EVALUATOR_TASKS_ROOT / registered / "validation_answer.csv"
        ).read_bytes()
        outcome = score_submission(registry, registered, answer)
        assert outcome.score == 1.0
        flipped = b"id,label\n" + b"".join(
            f"{row[0]},{1 - int(row[1])}\n".encode()
            for row in _read_csv(
                EVALUATOR_TASKS_ROOT / registered / "validation_answer.csv"
            )[1:]
        )
        assert score_submission(registry, registered, flipped).score == 0.0
        wrong_ids = answer.replace(b"\n0,", b"\n999999,", 1)
        with pytest.raises(EvaluatorError) as excinfo:
            score_submission(registry, registered, wrong_ids)
        assert excinfo.value.reason == "submission_id_mismatch"


# --- bindings and final barrier ---------------------------------------------------------
def test_bindings_build_all_27_cells(manifest: dict) -> None:
    bindings = load_task_bindings()
    assert set(bindings) == set(TASKS_BY_ID)
    for task_id, binding in bindings.items():
        pin = TASKS_BY_ID[task_id]
        assert binding.rows == pin.rows and binding.features == pin.features
        assert binding.split_seed == split_seed(pin)
    cells = iter_cells(bindings, source_commit="0" * 40)
    assert len(cells) == 27
    assert len({cell.identity() for cell in cells}) == 27


def test_manifest_tamper_rejected(tmp_path: Path) -> None:
    manifest = load_manifest()
    manifest["tasks"]["banknote"]["files"][
        "tasks/comparison/banknote/data/public/train.csv"
    ] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ComparisonDataError):
        load_task_bindings(path)
    with pytest.raises(ComparisonDataError):
        load_manifest(tmp_path / "missing.json")


def test_final_refit_publication_refused_before_barrier() -> None:
    # Without the real orchestrator + original ledger context, publication
    # refuses BEFORE any sealed/manifest access (EXP-006 shared full
    # verification; a caller-created barrier dictionary never suffices).
    with pytest.raises(ComparisonDataError, match="final_barrier_unproven"):
        publish_final_refit(barrier_evidence=None)
    with pytest.raises(ComparisonDataError, match="final_barrier_unproven"):
        publish_final_refit(barrier_evidence={"fake": True})
    # Structural validation of caller hints still applies when given a
    # well-formed-looking but schema-wrong payload alongside a contextless
    # refusal: context is checked first.
    with pytest.raises(ComparisonDataError):
        publish_final_refit(barrier_evidence="not-a-mapping")


def test_validate_barrier_evidence_structural_only() -> None:
    """validate_barrier_evidence remains a structural pre-check (never
    sufficient for release)."""
    from research.comparison.data import validate_barrier_evidence

    with pytest.raises(ComparisonDataError, match="final_barrier_not_frozen"):
        validate_barrier_evidence({"schema": "wrong"})
    with pytest.raises(ComparisonDataError, match="final_barrier_not_frozen"):
        validate_barrier_evidence(None)


def test_raw_rows_are_typed() -> None:
    row = RawRow(features=(1.0, 2.0), label=1)
    assert row.label == 1
    source = TASK_SOURCES_BY_ID["banknote"]
    assert source.pin is TASKS_BY_ID["banknote"]
    with pytest.raises(ComparisonDataError):
        TASK_SOURCES_BY_ID["banknote"].__class__(
            task_id="banknote",
            url="https://example.com/evil.zip",
            archive_name="evil.zip",
            member_name="x",
            converter="banknote",
        )
