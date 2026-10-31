"""Validation helpers for Instance objects and generated-data checksums."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from src.core.instance import Instance
from src.core.problem import is_feasible

logger = logging.getLogger(__name__)


def validate_instance(inst: Instance, check_planted: bool = True) -> None:
    """Check the dimensions and invariants of an Instance in CSR format."""
    # Validate basic problem dimensions
    if inst.n_exams < 2:
        raise ValueError(f"[{inst.name}] n_exams must be >= 2, got {inst.n_exams}")
    if inst.n_slots < 2 or inst.n_slots > inst.n_exams:
        raise ValueError(f"[{inst.name}] invalid n_slots={inst.n_slots}")
    if inst.n_students < 1:
        raise ValueError(f"[{inst.name}] n_students must be >= 1, got {inst.n_students}")
    if inst.n_edges < 0:
        raise ValueError(f"[{inst.name}] n_edges cannot be negative")

    arrays = {
        "indptr": inst.indptr,
        "indices": inst.indices,
        "weights": inst.weights,
        "exam_size": inst.exam_size,
    }
    # Loop over all 4 CSR arrays to verify they are NumPy arrays with dtype int32
    for array_name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            raise ValueError(f"[{inst.name}] {array_name} must be a NumPy array")
        if array.dtype != np.int32:
            raise ValueError(f"[{inst.name}] {array_name} must be int32, got {array.dtype}")

    # Check CSR row pointer array length, starting offset, and monotonicity
    if inst.indptr.shape != (inst.n_exams + 1,):
        raise ValueError(f"[{inst.name}] indptr shape mismatch: {inst.indptr.shape}")
    if int(inst.indptr[0]) != 0:
        raise ValueError(f"[{inst.name}] indptr[0] must be 0")
    if np.any(np.diff(inst.indptr) < 0):
        raise ValueError(f"[{inst.name}] indptr must be non-decreasing")

    # Check that indices and weights match indptr[-1] and equal 2 * n_edges
    nnz = int(inst.indptr[-1])
    if nnz < 0:
        raise ValueError(f"[{inst.name}] indptr[-1] cannot be negative")
    if inst.indices.shape != (nnz,) or inst.weights.shape != (nnz,):
        raise ValueError(f"[{inst.name}] indices/weights length must equal indptr[-1]={nnz}")
    if nnz != 2 * inst.n_edges:
        raise ValueError(f"[{inst.name}] nnz={nnz} does not equal 2 * n_edges={2 * inst.n_edges}")

    # Check exam_size bounds and edge weight bounds
    if inst.exam_size.shape != (inst.n_exams,):
        raise ValueError(f"[{inst.name}] exam_size shape mismatch: {inst.exam_size.shape}")
    if np.any(inst.exam_size < 0):
        raise ValueError(f"[{inst.name}] exam_size cannot contain negative values")
    if np.any(inst.exam_size > inst.n_students):
        raise ValueError(f"[{inst.name}] an exam enrolment cannot exceed n_students")
    if np.any(inst.weights <= 0):
        raise ValueError(f"[{inst.name}] all stored edge weights must be positive")
    if np.any(inst.weights > inst.n_students):
        raise ValueError(f"[{inst.name}] an edge weight cannot exceed n_students")

    edge_map: dict[tuple[int, int], int] = {}
    # Loop over every exam row to check sorted neighbor order and absence of self-loops
    for i in range(inst.n_exams):
        start, end = int(inst.indptr[i]), int(inst.indptr[i + 1])
        row_indices = inst.indices[start:end]
        if row_indices.size > 1 and np.any(np.diff(row_indices) <= 0):
            raise ValueError(
                f"[{inst.name}] neighbor indices in row {i} must be strictly increasing"
            )
        for offset in range(start, end):
            j = int(inst.indices[offset])
            weight = int(inst.weights[offset])
            if j < 0 or j >= inst.n_exams or j == i:
                raise ValueError(f"[{inst.name}] invalid neighbor index {j} in row {i}")
            edge_map[(i, j)] = weight

    # Verify undirected graph symmetry: every edge (i, j) has matching weight on (j, i)
    for (i, j), weight in edge_map.items():
        if edge_map.get((j, i)) != weight:
            raise ValueError(f"[{inst.name}] asymmetric edge ({i}, {j}) or weight mismatch")

    # If total_enrolments is stored in metadata, verify it matches exam_size.sum()
    if "total_enrolments" in inst.meta:
        try:
            recorded_total = int(inst.meta["total_enrolments"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"[{inst.name}] total_enrolments metadata must be an integer") from exc
        if recorded_total != int(inst.exam_size.sum()):
            raise ValueError(
                f"[{inst.name}] exam_size sum {int(inst.exam_size.sum())} does not match "
                f"metadata total_enrolments={recorded_total}"
            )

    # Optionally check that the planted reference timetable is feasible (V(x) == 0)
    if check_planted and "planted_solution" in inst.meta:
        try:
            planted = np.asarray(inst.meta["planted_solution"], dtype=np.int32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"[{inst.name}] planted_solution is not a valid integer vector"
            ) from exc
        if planted.shape != (inst.n_exams,):
            raise ValueError(
                f"[{inst.name}] planted_solution shape must be {(inst.n_exams,)}, "
                f"got {planted.shape}"
            )
        if not is_feasible(inst, planted):
            raise ValueError(f"[{inst.name}] planted_solution is not feasible")


def compute_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    # Read file in binary chunks to compute its SHA-256 hash
    with Path(file_path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(directory: str | Path, checksum_file: str | Path) -> dict[str, str]:
    """Write SHA-256 checksums for every .npz file in a directory."""
    folder = Path(directory)
    if not folder.is_dir():
        raise FileNotFoundError(f"Generated-data directory not found: {folder}")

    digests: dict[str, str] = {}
    lines: list[str] = []
    # Loop over sorted .npz files and record '<sha256>  <filename>'
    for npz_file in sorted(folder.glob("*.npz")):
        digest = compute_sha256(npz_file)
        digests[npz_file.name] = digest
        lines.append(f"{digest}  {npz_file.name}")

    out_path = Path(checksum_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return digests


def verify_checksums(directory: str | Path, checksum_file: str | Path) -> bool:
    """Verify that the checksum manifest exactly matches the .npz files present."""
    folder = Path(directory)
    manifest_path = Path(checksum_file)
    # Return False if either the directory or the manifest file is missing
    if not folder.is_dir() or not manifest_path.is_file():
        return False

    expected: dict[str, str] = {}
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False

    # Parse each line in checksums.txt
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            return False
        digest, filename = parts
        # Validate hash length, hex characters, and safe filename format
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
            or Path(filename).name != filename
            or not filename.endswith(".npz")
            or filename in expected
        ):
            return False
        expected[filename] = digest.lower()

    # Check that the set of .npz files on disk matches the manifest entries
    actual_names = {path.name for path in folder.glob("*.npz") if path.is_file()}
    if actual_names != set(expected):
        missing = sorted(set(expected) - actual_names)
        unexpected = sorted(actual_names - set(expected))
        logger.error(
            "Checksum manifest/file-set mismatch; missing=%s unexpected=%s",
            missing,
            unexpected,
        )
        return False

    # Compare the actual SHA-256 hash of each file against the manifest
    for filename, expected_digest in expected.items():
        target = folder / filename
        if compute_sha256(target) != expected_digest:
            logger.error("Checksum mismatch for %s", target)
            return False
    return True
