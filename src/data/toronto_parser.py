"""Reader for Carter Toronto benchmark .crs and .stu files."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.core.instance import Instance

logger = logging.getLogger(__name__)


def build_csr_from_symmetric_matrix(
    conflict_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Convert a symmetric conflict matrix to CSR arrays."""
    matrix = np.asarray(conflict_matrix)
    # Check that the input is a 2D square matrix
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("conflict_matrix must be a square matrix")
    # Check that entries are integer counts
    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("conflict_matrix weights must be integers")
    # Reject negative counts or values overflowing int32
    if np.any(matrix < 0) or np.any(matrix > np.iinfo(np.int32).max):
        raise ValueError("conflict_matrix weights must fit in int32 and be nonnegative")
    # Ensure no exam conflicts with itself on the diagonal
    if np.any(np.diag(matrix) != 0):
        raise ValueError("conflict_matrix diagonal must be zero")
    # Ensure c_ij == c_ji across the whole matrix
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("conflict_matrix must be symmetric")

    n_exams = int(matrix.shape[0])
    indptr = np.zeros(n_exams + 1, dtype=np.int32)
    indices_parts: list[np.ndarray] = []
    weights_parts: list[np.ndarray] = []

    # Loop over each row to collect non-zero neighbors and edge weights
    for i in range(n_exams):
        neighbors = np.flatnonzero(matrix[i] > 0).astype(np.int32)
        indices_parts.append(neighbors)
        weights_parts.append(matrix[i, neighbors].astype(np.int32, copy=True))
        indptr[i + 1] = indptr[i] + neighbors.size

    indices = (
        np.concatenate(indices_parts).astype(np.int32, copy=False)
        if indices_parts
        else np.empty(0, dtype=np.int32)
    )
    weights = (
        np.concatenate(weights_parts).astype(np.int32, copy=False)
        if weights_parts
        else np.empty(0, dtype=np.int32)
    )
    return indptr, indices, weights, int(indices.size // 2)


def parse_crs_file(crs_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read course IDs and header enrolment counts from a Toronto .crs file."""
    path = Path(crs_path)
    # Verify that the .crs file exists on disk
    if not path.is_file():
        raise FileNotFoundError(f"Toronto .crs file not found: {path}")

    exam_ids: list[str] = []
    sizes: list[int] = []
    seen_ids: set[str] = set()

    # Loop over lines in the .crs file to extract exam ID and student count
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        # Skip empty lines and comment lines
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Each valid line must have exactly 2 fields: <exam_id> <enrolment>
        if len(parts) != 2:
            raise ValueError(
                f"Malformed .crs record at {path}:{line_number}; expected '<exam_id> <enrolment>'"
            )
        exam_id = parts[0]
        try:
            enrolment = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid enrolment count at {path}:{line_number}: {parts[1]!r}"
            ) from exc
        # Reject duplicate course IDs
        if exam_id in seen_ids:
            raise ValueError(f"Duplicate exam ID {exam_id!r} in {path}:{line_number}")
        # Reject negative enrollment numbers
        if enrolment < 0:
            raise ValueError(f"Negative enrolment count in {path}:{line_number}")

        seen_ids.add(exam_id)
        exam_ids.append(exam_id)
        sizes.append(enrolment)

    # Ensure the file contained at least one course
    if not exam_ids:
        raise ValueError(f"No course entries found in {path}")
    return exam_ids, np.asarray(sizes, dtype=np.int32)


def parse_toronto_instance(
    name: str,
    crs_path: str | Path,
    stu_path: str | Path,
    n_slots: int,
    *,
    strict_enrolment: bool = False,
) -> Instance:
    """Parse a Toronto .crs/.stu pair into the shared CSR Instance type."""
    # Require at least 2 time slots
    if n_slots < 2:
        raise ValueError(f"n_slots must be at least 2, got {n_slots}")

    # Parse course list and map each string exam ID to a 0-based integer index
    exam_ids, crs_sizes = parse_crs_file(crs_path)
    id_to_idx = {exam_id: idx for idx, exam_id in enumerate(exam_ids)}
    n_exams = len(exam_ids)
    if n_slots > n_exams:
        raise ValueError(f"n_slots={n_slots} cannot exceed n_exams={n_exams}")

    stu_file = Path(stu_path)
    # Verify that the .stu file exists
    if not stu_file.is_file():
        raise FileNotFoundError(f"Toronto .stu file not found: {stu_file}")

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    actual_sizes = np.zeros(n_exams, dtype=np.int32)
    n_students = 0

    student_lines = stu_file.read_text(encoding="utf-8-sig").splitlines()
    # Loop over each student line in the .stu file
    for line_number, raw_line in enumerate(student_lines, 1):
        line = raw_line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        # Check if any course code on this line is missing from the .crs file
        unknown = sorted(set(tokens).difference(id_to_idx))
        if unknown:
            raise ValueError(
                f"Unknown exam ID(s) {unknown} at {stu_file}:{line_number}; "
                "all IDs must appear in the .crs file"
            )

        # Convert tokens to sorted unique exam indices for this student
        enrolled = sorted({id_to_idx[token] for token in tokens})
        if not enrolled:
            continue
        n_students += 1

        # Increment enrollment count for each exam taken by this student
        for exam_idx in enrolled:
            actual_sizes[exam_idx] += 1

        # Loop over all unordered exam pairs for this student to count conflicts
        for left in range(len(enrolled)):
            i = enrolled[left]
            for right in range(left + 1, len(enrolled)):
                j = enrolled[right]
                pair_counts[(i, j)] += 1

    # Check if header counts in .crs differ from actual student counts in .stu
    discrepancies = {
        exam_ids[i]: {"crs": int(crs_sizes[i]), "stu": int(actual_sizes[i])}
        for i in range(n_exams)
        if int(crs_sizes[i]) != int(actual_sizes[i])
    }
    if discrepancies:
        message = (
            f"Enrolment mismatch between {Path(crs_path).name} and "
            f"{stu_file.name} for {len(discrepancies)} exam(s)"
        )
        # Raise error in strict mode, otherwise log a warning and use .stu counts
        if strict_enrolment:
            raise ValueError(f"{message}: {discrepancies}")
        logger.warning("%s; using counts from the student file", message)

    # Build symmetric adjacency lists from pairwise conflict counts
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_exams)]
    for (i, j), weight in pair_counts.items():
        if weight > 0:
            adjacency[i].append((j, weight))
            adjacency[j].append((i, weight))

    # Convert adjacency lists into sorted CSR arrays
    indptr = np.zeros(n_exams + 1, dtype=np.int32)
    indices_parts: list[np.ndarray] = []
    weights_parts: list[np.ndarray] = []
    for i, neighbors in enumerate(adjacency):
        neighbors.sort(key=lambda item: item[0])
        indices_parts.append(np.fromiter((j for j, _ in neighbors), dtype=np.int32))
        weights_parts.append(np.fromiter((w for _, w in neighbors), dtype=np.int32))
        indptr[i + 1] = indptr[i] + len(neighbors)

    indices = (
        np.concatenate(indices_parts).astype(np.int32, copy=False)
        if indices_parts
        else np.empty(0, dtype=np.int32)
    )
    weights = (
        np.concatenate(weights_parts).astype(np.int32, copy=False)
        if weights_parts
        else np.empty(0, dtype=np.int32)
    )
    n_edges = len(pair_counts)

    logger.info(
        "Parsed Toronto instance '%s': N=%d, T=%d, S=%d, |E|=%d",
        name,
        n_exams,
        n_slots,
        n_students,
        n_edges,
    )
    return Instance(
        name=name,
        n_exams=n_exams,
        n_students=n_students,
        n_slots=int(n_slots),
        indptr=indptr,
        indices=indices,
        weights=weights,
        exam_size=actual_sizes,
        n_edges=n_edges,
        meta={
            "source": "toronto",
            "crs_file": Path(crs_path).name,
            "stu_file": stu_file.name,
            "header_enrolment_discrepancies": discrepancies,
        },
    )
