"""Synthetic examination-timetabling instances with a planted feasible timetable."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.core.instance import Instance
from src.core.problem import proximity_cost

logger = logging.getLogger(__name__)


def generate_instance(
    name: str,
    n_exams: int,
    n_slots: int,
    n_students: int,
    seed: int,
    courses_per_student: int = 4,
    n_cohorts: int = 3,
) -> Instance:
    """Generate a reproducible synthetic instance with a feasible planted timetable.

    Args:
        name: Identifier stored with the instance.
        n_exams: Number of exams, N.
        n_slots: Number of available time slots, T.
        n_students: Number of synthetic student records, S (must be >= N).
        seed: Seed passed to np.random.default_rng.
        courses_per_student: Target average number of exams per student.
        n_cohorts: Number of academic cohorts used to group exams.

    Returns:
        An Instance whose conflict graph is stored in CSR format.
        Time complexity: O(S * k^2 + N^2).
    """
    # Check that there are at least 2 exams to form a valid problem
    if n_exams < 2:
        raise ValueError(f"n_exams must be at least 2, got {n_exams}")
    # Check that the number of slots is between 2 and n_exams
    if n_slots < 2 or n_slots > n_exams:
        raise ValueError(f"n_slots must satisfy 2 <= n_slots <= n_exams; got {n_slots}")
    # Require enough students so every exam can be taken by at least one student
    if n_students < n_exams:
        raise ValueError(
            "n_students must be at least n_exams so the generator can cover every exam"
        )
    # Check that a student's course load does not exceed total available time slots
    if not 2 <= courses_per_student <= n_slots:
        raise ValueError("courses_per_student must satisfy 2 <= courses_per_student <= n_slots")
    # Require at least one student cohort
    if n_cohorts < 1:
        raise ValueError(f"n_cohorts must be positive, got {n_cohorts}")

    # Create an isolated random number generator seeded for reproducibility
    rng = np.random.default_rng(seed)
    n_cohorts = min(int(n_cohorts), n_exams)

    # Assign exams evenly across slots and shuffle to create the planted solution
    planted_x = np.arange(n_exams, dtype=np.int32) % n_slots
    rng.shuffle(planted_x)

    # Assign exams across academic cohorts and shuffle
    exam_cohort = np.arange(n_exams, dtype=np.int32) % n_cohorts
    rng.shuffle(exam_cohort)

    # Draw exam popularity weights from a Gamma distribution
    exam_popularity = rng.gamma(shape=2.0, scale=1.0, size=n_exams)

    # Group exam indices by their assigned planted time slot
    slot_to_exams = [np.flatnonzero(planted_x == slot).astype(np.int32) for slot in range(n_slots)]

    # Precompute slot and exam sampling probabilities for each cohort
    cohort_slot_weights = np.empty((n_cohorts, n_slots), dtype=np.float64)
    selection_probabilities: dict[tuple[int, int], np.ndarray] = {}

    # Loop over each cohort to build its probability tables
    for cohort_id in range(n_cohorts):
        cohort_mask = exam_cohort == cohort_id
        # Loop over each time slot to weight exams belonging to this cohort
        for slot in range(n_slots):
            candidates = slot_to_exams[slot]
            in_cohort = int(np.count_nonzero(cohort_mask[candidates]))
            cohort_slot_weights[cohort_id, slot] = 5.0 * in_cohort + 0.5

            candidate_weights = exam_popularity[candidates].copy()
            # Give higher selection weight to exams from the student's own cohort
            candidate_weights *= np.where(exam_cohort[candidates] == cohort_id, 8.0, 1.0)
            candidate_weights /= candidate_weights.sum()
            selection_probabilities[(cohort_id, slot)] = candidate_weights

        # Normalize slot weights so they sum to 1.0 for rng.choice
        cohort_slot_weights[cohort_id] /= cohort_slot_weights[cohort_id].sum()

    # Initialize dense conflict matrix and per-exam enrollment counters
    conflict_matrix = np.zeros((n_exams, n_exams), dtype=np.int32)
    exam_size = np.zeros(n_exams, dtype=np.int32)
    total_enrolments = 0

    # Set lower and upper bounds for how many exams each student takes
    min_k = max(2, min(n_slots, courses_per_student - 1))
    max_k = min(n_slots, courses_per_student + 1)
    # Randomize the order in which the first N students cover all N exams
    coverage_order = rng.permutation(n_exams)

    # Loop over all students to simulate course enrollments
    for student_idx in range(n_students):
        cohort_id = student_idx % n_cohorts
        # Draw number of exams k for this student
        k = int(rng.integers(min_k, max_k + 1))
        # Pick k distinct time slots so no two chosen exams share a planted slot
        chosen_slots = rng.choice(
            n_slots,
            size=k,
            replace=False,
            p=cohort_slot_weights[cohort_id],
        )

        # For the first N students, force coverage of one distinct exam per student
        target_exam = int(coverage_order[student_idx]) if student_idx < n_exams else -1
        target_slot = int(planted_x[target_exam]) if target_exam >= 0 else -1
        # If the target exam's slot wasn't picked, replace the last picked slot with it
        if target_exam >= 0 and not np.any(chosen_slots == target_slot):
            chosen_slots[-1] = target_slot

        chosen_exams = np.empty(k, dtype=np.int32)
        # Pick one exam from each chosen time slot
        for idx, slot_value in enumerate(chosen_slots):
            slot = int(slot_value)
            # Use the forced target exam if this is its slot, otherwise sample by weight
            if target_exam >= 0 and slot == target_slot:
                chosen_exams[idx] = target_exam
            else:
                candidates = slot_to_exams[slot]
                chosen_exams[idx] = int(
                    rng.choice(
                        candidates,
                        p=selection_probabilities[(cohort_id, slot)],
                    )
                )

        # Sort chosen exams and update enrollment counts
        chosen_exams.sort()
        exam_size[chosen_exams] += 1
        total_enrolments += k

        # Loop over all pairs of exams taken by this student to increment c_ij
        for left in range(k):
            i = int(chosen_exams[left])
            for right in range(left + 1, k):
                j = int(chosen_exams[right])
                conflict_matrix[i, j] += 1
                conflict_matrix[j, i] += 1

    # Sanity-check that total enrollments match the sum of exam sizes
    if int(exam_size.sum()) != total_enrolments:
        raise RuntimeError("Generated enrolment totals do not match student records")
    # Sanity-check that no exam was left with zero students
    if np.any(exam_size == 0):
        missing = np.flatnonzero(exam_size == 0).tolist()
        raise RuntimeError(f"Generator failed to cover exams: {missing}")

    # Convert the dense symmetric conflict matrix into CSR arrays
    indptr = np.zeros(n_exams + 1, dtype=np.int32)
    indices_parts: list[np.ndarray] = []
    weights_parts: list[np.ndarray] = []

    # Loop through each exam row to extract non-zero conflict edges
    for exam_idx in range(n_exams):
        neighbors = np.flatnonzero(conflict_matrix[exam_idx] > 0).astype(np.int32)
        edge_weights = conflict_matrix[exam_idx, neighbors].astype(np.int32, copy=True)
        indices_parts.append(neighbors)
        weights_parts.append(edge_weights)
        indptr[exam_idx + 1] = indptr[exam_idx] + neighbors.size

    # Concatenate row slices into flat CSR arrays
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
    n_edges = int(indices.size // 2)

    # Build the immutable Instance object
    inst = Instance(
        name=name,
        n_exams=int(n_exams),
        n_students=int(n_students),
        n_slots=int(n_slots),
        indptr=indptr,
        indices=indices,
        weights=weights,
        exam_size=exam_size,
        n_edges=n_edges,
        meta={
            "source": "synthetic",
            "generator_version": "planted-cohort-v2",
            "seed": int(seed),
            "courses_per_student": int(courses_per_student),
            "n_cohorts": int(n_cohorts),
            "planted_solution": planted_x.tolist(),
            "planted_solution_usage": "validation_only",
            "total_enrolments": int(total_enrolments),
        },
    )
    # Compute and store the Carter proximity cost of the planted timetable
    inst.meta["planted_proximity_cost"] = round(float(proximity_cost(inst, planted_x)), 6)
    return inst


def save_instance_npz(inst: Instance, path: str | Path) -> Path:
    """Save an instance to a compressed .npz file."""
    out_path = Path(path)
    # Ensure the parent directory exists before saving
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize the metadata dictionary as a JSON string
    meta_json = json.dumps(inst.meta, sort_keys=True)
    # Write all CSR arrays and dimensions into a compressed archive
    np.savez_compressed(
        out_path,
        name=np.array(inst.name),
        n_exams=np.array(inst.n_exams, dtype=np.int32),
        n_students=np.array(inst.n_students, dtype=np.int32),
        n_slots=np.array(inst.n_slots, dtype=np.int32),
        indptr=inst.indptr,
        indices=inst.indices,
        weights=inst.weights,
        exam_size=inst.exam_size,
        n_edges=np.array(inst.n_edges, dtype=np.int32),
        meta_json=np.array(meta_json),
    )
    return out_path


def load_instance_npz(path: str | Path) -> Instance:
    """Load an instance from a compressed .npz file without enabling pickle."""
    in_path = Path(path)
    # Open the .npz archive safely with allow_pickle=False
    with np.load(in_path, allow_pickle=False) as data:
        required = {
            "name",
            "n_exams",
            "n_students",
            "n_slots",
            "indptr",
            "indices",
            "weights",
            "exam_size",
            "n_edges",
            "meta_json",
        }
        missing = required.difference(data.files)
        # Raise an error if any expected array is missing from the archive
        if missing:
            raise ValueError(f"{in_path} is missing required arrays: {sorted(missing)}")
        # Parse the JSON metadata string back into a Python dict
        try:
            meta: dict[str, Any] = json.loads(str(data["meta_json"].item()))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid metadata JSON in {in_path}") from exc

        # Reconstruct and return the immutable Instance object
        return Instance(
            name=str(data["name"].item()),
            n_exams=int(data["n_exams"].item()),
            n_students=int(data["n_students"].item()),
            n_slots=int(data["n_slots"].item()),
            indptr=np.array(data["indptr"], dtype=np.int32),
            indices=np.array(data["indices"], dtype=np.int32),
            weights=np.array(data["weights"], dtype=np.int32),
            exam_size=np.array(data["exam_size"], dtype=np.int32),
            n_edges=int(data["n_edges"].item()),
            meta=meta,
        )


def generate_all_from_yaml(
    yaml_path: str | Path,
    output_dir: str | Path,
) -> list[tuple[Instance, Path]]:
    """Generate every synthetic instance defined in data/instances.yaml."""
    config_path = Path(yaml_path)
    # Load the YAML configuration file
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    groups = config.get("groups", {})
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Instance, Path]] = []
    # Loop over each benchmark group (tiny, small, medium, large, tuning)
    for group_name, group_spec in groups.items():
        # Skip external Toronto benchmarks since they are parsed from files, not generated
        if group_name == "toronto" or group_spec.get("optional", False):
            continue

        defaults = {
            "n_exams": group_spec.get("n_exams"),
            "n_slots": group_spec.get("n_slots"),
            "n_students": group_spec.get("n_students"),
            "courses_per_student": group_spec.get("courses_per_student", 4),
            "n_cohorts": group_spec.get("n_cohorts", 3),
        }
        # Loop over each instance entry in the current group
        for item in group_spec.get("instances", []):
            params: dict[str, int] = {}
            # Merge instance-level overrides with group-level defaults
            for key, default in defaults.items():
                value = item.get(key, default)
                if value is None:
                    raise ValueError(
                        f"Missing required parameter {key!r} for instance {item.get('name')!r}"
                    )
                params[key] = int(value)
            # Ensure every synthetic instance has a fixed random seed
            if "seed" not in item:
                raise ValueError(f"Missing seed for instance {item.get('name')!r}")

            # Call generate_instance and save the resulting .npz file
            inst = generate_instance(
                name=str(item["name"]),
                seed=int(item["seed"]),
                **params,
            )
            inst.meta["group"] = str(group_name)
            inst.meta["budget_fee"] = int(
                item.get("budget_fee", group_spec.get("budget_fee", 50000))
            )
            output_path = save_instance_npz(inst, out_root / f"{inst.name}.npz")
            logger.info("Generated %s -> %s", inst.name, output_path.name)
            results.append((inst, output_path))

    return results
