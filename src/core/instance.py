"""Problem instance representation backed by a CSR conflict graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Instance:
    """Immutable examination timetabling instance stored in CSR format.

    Attributes:
        name: Instance identifier (e.g., 'small-a', 'car-s-91').
        n_exams: Number of exams N.
        n_students: Total number of students S.
        n_slots: Number of available time slots T.
        indptr: CSR row pointers of shape (N + 1,), dtype int32.
        indices: CSR column indices (neighbour exam IDs), dtype int32.
        weights: Shared student counts c_ij aligned with `indices`, dtype int32.
        exam_size: Total student enrolment per exam of shape (N,), dtype int32.
        n_edges: Number of undirected conflict edges |E| with c_ij > 0.
        meta: Optional metadata dictionary (source, seed, generator settings).
    """

    name: str
    n_exams: int
    n_students: int
    n_slots: int
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    exam_size: np.ndarray
    n_edges: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize array dtypes to int32, validate CSR shapes, and lock buffers.

        Raises:
            ValueError: If dimensions or CSR pointer bounds are inconsistent.

        Time Complexity:
            O(1) when arrays are already int32; O(N + |E|) if dtype casting is needed.
        """
        if self.n_exams <= 0:
            raise ValueError(f"n_exams must be > 0, got {self.n_exams}.")
        if self.n_students <= 0:
            raise ValueError(f"n_students must be > 0, got {self.n_students}.")
        if self.n_slots <= 0:
            raise ValueError(f"n_slots must be > 0, got {self.n_slots}.")
        if self.n_edges < 0:
            raise ValueError(f"n_edges must be >= 0, got {self.n_edges}.")

        for attr in ("indptr", "indices", "weights", "exam_size"):
            raw = getattr(self, attr)
            arr = np.asarray(raw, dtype=np.int32)
            if arr.ndim != 1:
                raise ValueError(f"'{attr}' must be a 1D array, got {arr.ndim}D.")
            arr.flags.writeable = False
            object.__setattr__(self, attr, arr)

        if self.indptr.shape[0] != self.n_exams + 1:
            raise ValueError(
                f"indptr length must be n_exams + 1 ({self.n_exams + 1}), "
                f"got {self.indptr.shape[0]}."
            )
        if self.indices.shape[0] != self.weights.shape[0]:
            raise ValueError(
                f"indices ({self.indices.shape[0]}) and weights "
                f"({self.weights.shape[0]}) must have the same length."
            )
        if self.exam_size.shape[0] != self.n_exams:
            raise ValueError(
                f"exam_size length must be n_exams ({self.n_exams}), "
                f"got {self.exam_size.shape[0]}."
            )
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != self.indices.shape[0]:
            raise ValueError(
                "CSR indptr bounds invalid: expected indptr[0] == 0 and "
                "indptr[-1] == len(indices)."
            )

    def degree(self, exam_i: int) -> int:
        """Return the number of conflicting neighbours deg(i) for exam `exam_i`.

        Args:
            exam_i: Exam index in [0, n_exams - 1].

        Returns:
            Number of exams sharing at least one student with `exam_i`.

        Time Complexity:
            O(1).
        """
        return int(self.indptr[exam_i + 1] - self.indptr[exam_i])

    def neighbors(self, exam_i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return neighbour indices and conflict weights for `exam_i`.

        Args:
            exam_i: Exam index in [0, n_exams - 1].

        Returns:
            Tuple of read-only slices `(neighbour_exams, conflict_counts)`.

        Time Complexity:
            O(1).
        """
        start = self.indptr[exam_i]
        end = self.indptr[exam_i + 1]
        return self.indices[start:end], self.weights[start:end]