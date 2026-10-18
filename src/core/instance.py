"""Instance data structure for the Examination Timetabling Problem (Section 3.2)."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Instance:
    """Problem instance representing the conflict graph in CSR format.

    The dataclass prevents reassignment of fields, but NumPy arrays and the metadata
    dictionary remain mutable unless callers treat them as read-only.

    Attributes:
        name: Instance identifier (e.g., 'tiny-a', 'car-s-91').
        n_exams: Total number of exams N.
        n_students: Total number of students S.
        n_slots: Number of available time slots T.
        indptr: CSR row pointer array of shape (N + 1,), dtype int32.
        indices: CSR column indices of conflicting exams, dtype int32.
        weights: Shared student counts c_ij aligned with indices, dtype int32.
        exam_size: Number of enrolled students per exam of shape (N,), dtype int32.
        n_edges: Number of unordered conflicting pairs |E| with c_ij > 0.
        meta: Dictionary storing source, generator parameters, seed, etc.
    """

    name: str
    n_exams: int
    n_students: int
    n_slots: int
    indptr: np.ndarray  # int32, len N+1 (conflict graph in CSR form)
    indices: np.ndarray  # int32, neighbours j of each exam i
    weights: np.ndarray  # int32, c_ij aligned with `indices`
    exam_size: np.ndarray  # int32, enrolment per exam
    n_edges: int  # |E| = number of unordered pairs with c_ij > 0
    meta: dict[str, Any] = field(default_factory=dict)

    def degree(self, exam_i: int) -> int:
        """Return the number of conflicting neighbours for exam_i in O(1) time."""
        return int(self.indptr[exam_i + 1] - self.indptr[exam_i])
