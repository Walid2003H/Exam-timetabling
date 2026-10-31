"""Tests for synthetic generation, Toronto parsing, persistence, and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.instance import Instance
from src.core.problem import is_feasible
from src.data import (
    generate_instance,
    load_instance,
    load_instance_npz,
    parse_toronto_instance,
    save_instance_npz,
    summarize_instances,
    validate_instance,
    verify_checksums,
    write_checksums,
)

ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_generator_is_reproducible_and_has_a_feasible_timetable() -> None:
    """A fixed seed must recreate the graph and its planted feasible schedule."""
    # Generate two instances with the same seed and one with a different seed
    first = generate_instance("test-small", n_exams=25, n_slots=8, n_students=200, seed=1000)
    second = generate_instance("test-small", n_exams=25, n_slots=8, n_students=200, seed=1000)
    other_seed = generate_instance("test-small", n_exams=25, n_slots=8, n_students=200, seed=1001)

    # Validate CSR structure and check reproducibility
    validate_instance(first, check_planted=True)
    assert np.array_equal(first.indptr, second.indptr)
    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.weights, second.weights)
    assert not np.array_equal(first.weights, other_seed.weights)
    assert np.all(first.exam_size > 0)
    assert int(first.exam_size.sum()) == first.meta["total_enrolments"]

    # Verify that the planted solution has zero hard conflicts
    planted = np.asarray(first.meta["planted_solution"], dtype=np.int32)
    assert is_feasible(first, planted)
    assert first.meta["planted_solution_usage"] == "validation_only"


def test_generator_rejects_inconsistent_dimensions() -> None:
    """Invalid generator settings should fail early with a ValueError."""
    # Check that fewer students than exams raises ValueError
    for bad_kwargs in [
        {"name": "too-few", "n_exams": 12, "n_slots": 6, "n_students": 10, "seed": 1},
        {
            "name": "bad-load",
            "n_exams": 12,
            "n_slots": 6,
            "n_students": 60,
            "seed": 1,
            "courses_per_student": 7,
        },
        {
            "name": "bad-cohorts",
            "n_exams": 12,
            "n_slots": 6,
            "n_students": 60,
            "seed": 1,
            "n_cohorts": 0,
        },
    ]:
        raised = False
        try:
            generate_instance(**bad_kwargs)
        except ValueError:
            raised = True
        assert raised is True


def test_npz_roundtrip_preserves_instance(tmp_path: Path | None = None) -> None:
    """Serializing an instance must preserve its graph and metadata."""
    folder = tmp_path if tmp_path is not None else Path("/tmp")
    instance = generate_instance("tiny-roundtrip", n_exams=12, n_slots=6, n_students=60, seed=1000)
    # Save to .npz and load back
    archive_path = save_instance_npz(instance, folder / "tiny-roundtrip.npz")
    loaded = load_instance_npz(archive_path)

    # Check that all attributes and CSR arrays match the original instance
    validate_instance(loaded, check_planted=True)
    assert loaded.name == instance.name
    assert loaded.n_exams == instance.n_exams
    assert loaded.n_slots == instance.n_slots
    assert loaded.n_students == instance.n_students
    assert loaded.n_edges == instance.n_edges
    assert loaded.meta == instance.meta
    for field in ("indptr", "indices", "weights", "exam_size"):
        assert np.array_equal(getattr(loaded, field), getattr(instance, field))


def test_toronto_parser_uses_student_records_and_reports_header_mismatches() -> None:
    """The sample Toronto pair should parse and expose discrepancies in its toy header."""
    crs_path = ROOT / "data" / "toronto" / "sample_toy.crs"
    stu_path = ROOT / "data" / "toronto" / "sample_toy.stu"

    # Parse in default mode (logs warning and records discrepancy in metadata)
    instance = parse_toronto_instance("sample-toy", crs_path, stu_path, n_slots=4)
    validate_instance(instance, check_planted=False)
    assert instance.n_exams == 6
    assert instance.n_students == 8
    assert instance.n_slots == 4
    assert instance.n_edges > 0
    assert instance.meta["header_enrolment_discrepancies"]
    assert np.array_equal(instance.exam_size, np.array([3, 4, 3, 2, 3, 3], dtype=np.int32))

    # Verify that strict_enrolment=True raises ValueError on header mismatch
    raised = False
    try:
        parse_toronto_instance(
            "sample-toy-strict", crs_path, stu_path, n_slots=4, strict_enrolment=True
        )
    except ValueError:
        raised = True
    assert raised is True


def test_toronto_parser_rejects_unknown_exam_ids(tmp_path: Path | None = None) -> None:
    """Unknown tokens must not be silently discarded from a student record."""
    folder = tmp_path if tmp_path is not None else Path("/tmp")
    crs_path = folder / "small.crs"
    stu_path = folder / "small.stu"
    crs_path.write_text("A 1\nB 1\n", encoding="utf-8")
    stu_path.write_text("A UNKNOWN\n", encoding="utf-8")

    raised = False
    try:
        parse_toronto_instance("invalid", crs_path, stu_path, n_slots=2)
    except ValueError:
        raised = True
    assert raised is True


def test_validator_rejects_asymmetric_graph_and_negative_exam_size() -> None:
    """The validator must identify a broken symmetry relation and invalid enrolment."""
    valid = generate_instance("validation-source", 8, 4, 20, seed=4)
    # Perturb one weight to break symmetry (c_ij != c_ji)
    bad_weights = valid.weights.copy()
    bad_weights[0] += 1
    asymmetric = Instance(
        name="asymmetric",
        n_exams=valid.n_exams,
        n_students=valid.n_students,
        n_slots=valid.n_slots,
        indptr=valid.indptr,
        indices=valid.indices,
        weights=bad_weights,
        exam_size=valid.exam_size,
        n_edges=valid.n_edges,
        meta={},
    )
    raised_asym = False
    try:
        validate_instance(asymmetric, check_planted=False)
    except ValueError:
        raised_asym = True
    assert raised_asym is True

    # Set a negative exam size and verify that validate_instance rejects it
    negative_size = valid.exam_size.copy()
    negative_size[0] = -1
    invalid_size = Instance(
        name="negative-size",
        n_exams=valid.n_exams,
        n_students=valid.n_students,
        n_slots=valid.n_slots,
        indptr=valid.indptr,
        indices=valid.indices,
        weights=valid.weights,
        exam_size=negative_size,
        n_edges=valid.n_edges,
        meta={},
    )
    raised_neg = False
    try:
        validate_instance(invalid_size, check_planted=False)
    except ValueError:
        raised_neg = True
    assert raised_neg is True


def test_generated_benchmark_suite_and_checksums() -> None:
    """All committed generated instances must load, validate, and match the manifest."""
    generated_dir = ROOT / "data" / "generated"
    checksum_file = ROOT / "data" / "checksums.txt"
    assert verify_checksums(generated_dir, checksum_file) is True

    expected_names = [
        "tiny-a",
        "tiny-b",
        "tiny-c",
        "small-a",
        "small-b",
        "small-c",
        "medium-a",
        "medium-b",
        "medium-c",
        "large-a",
        "large-b",
        "large-c",
        "tune-small",
        "tune-medium",
        "tune-large",
    ]
    assert {path.stem for path in generated_dir.glob("*.npz")} == set(expected_names)

    # Load and validate every generated instance
    loaded = [load_instance(name, generated_dir=generated_dir) for name in expected_names]
    for instance in loaded:
        validate_instance(instance, check_planted=True)
        assert np.all(instance.exam_size > 0)
        assert int(instance.exam_size.sum()) == instance.meta["total_enrolments"]

    summary = summarize_instances(loaded)
    assert len(summary) == 15
    assert set(summary["name"]) == set(expected_names)


def test_checksum_verification_detects_unlisted_npz(tmp_path: Path | None = None) -> None:
    """An extra archive must invalidate the manifest rather than go unnoticed."""
    folder = tmp_path if tmp_path is not None else Path("/tmp/chk_test")
    generated_dir = folder / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for old in generated_dir.glob("*.npz"):
        old.unlink()

    np.savez_compressed(generated_dir / "one.npz", value=np.array([1]))
    manifest = folder / "checksums.txt"
    write_checksums(generated_dir, manifest)
    assert verify_checksums(generated_dir, manifest) is True

    # Add an unlisted .npz file and verify that checksum check returns False
    np.savez_compressed(generated_dir / "unlisted.npz", value=np.array([2]))
    assert verify_checksums(generated_dir, manifest) is False


if __name__ == "__main__":
    test_synthetic_generator_is_reproducible_and_has_a_feasible_timetable()
    test_generator_rejects_inconsistent_dimensions()
    test_npz_roundtrip_preserves_instance()
    test_toronto_parser_uses_student_records_and_reports_header_mismatches()
    test_toronto_parser_rejects_unknown_exam_ids()
    test_validator_rejects_asymmetric_graph_and_negative_exam_size()
    test_generated_benchmark_suite_and_checksums()
    test_checksum_verification_detects_unlisted_npz()
    print("All 8 P2 data unit tests passed!")
