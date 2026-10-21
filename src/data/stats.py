"""Summary statistics computation for examination timetabling instances."""

from typing import Any

import numpy as np
import pandas as pd

from src.core.instance import Instance


def compute_instance_stats(inst: Instance) -> dict[str, Any]:
    """Compute key structural metrics for a single Instance."""
    # Compute vertex degrees in O(N) by taking differences of consecutive CSR pointers
    degrees = np.diff(inst.indptr).astype(np.float64)
    # Compute maximum possible undirected edges N*(N-1)/2 for graph density
    max_possible_edges = inst.n_exams * (inst.n_exams - 1) // 2
    density = float(inst.n_edges) / float(max(1, max_possible_edges))

    # Compute mean and maximum conflict weight c_ij if edges exist
    w_mean = float(np.mean(inst.weights)) if inst.weights.size > 0 else 0.0
    w_max = int(np.max(inst.weights)) if inst.weights.size > 0 else 0

    return {
        "name": inst.name,
        "group": str(inst.meta.get("group", inst.meta.get("source", "unknown"))),
        "n_exams": inst.n_exams,
        "n_slots": inst.n_slots,
        "n_students": inst.n_students,
        "n_edges": inst.n_edges,
        "conflict_density": round(density, 4),
        "deg_min": int(np.min(degrees)),
        "deg_mean": round(float(np.mean(degrees)), 2),
        "deg_max": int(np.max(degrees)),
        "deg_std": round(float(np.std(degrees)), 2),
        "enrol_min": int(np.min(inst.exam_size)),
        "enrol_mean": round(float(np.mean(inst.exam_size)), 2),
        "enrol_max": int(np.max(inst.exam_size)),
        "weight_mean": round(w_mean, 2),
        "weight_max": w_max,
        "planted_cost": inst.meta.get("planted_proximity_cost", None),
    }


def summarize_instances(instances: list[Instance]) -> pd.DataFrame:
    """Build a pandas DataFrame summarizing a list of instances."""
    # Collect stats dictionaries for all instances and return as a DataFrame
    rows = [compute_instance_stats(inst) for inst in instances]
    return pd.DataFrame(rows)
