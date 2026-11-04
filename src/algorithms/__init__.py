"""Metaheuristic solvers for the Examination Timetabling Problem (P5 GA, P6 PSO, P7 ACO)."""

from __future__ import annotations

__all__: list[str] = []

try:
    from src.algorithms.ga import GAMemeticSolver, GASolver

    __all__.extend(["GAMemeticSolver", "GASolver"])
except ImportError:  # pragma: no cover
    pass

try:
    from src.algorithms.pso import (
        PSOMemeticSolver,
        PSOSolver,
        decode_random_keys,
        encode_slots_to_keys,
        update_direct_discrete_particle,
    )

    __all__.extend(
        [
            "PSOMemeticSolver",
            "PSOSolver",
            "decode_random_keys",
            "encode_slots_to_keys",
            "update_direct_discrete_particle",
        ]
    )
except ImportError:  # pragma: no cover
    pass

try:
    from src.algorithms.aco import (
        ACOMemeticSolver,
        ACOSolver,
        compute_mmas_bounds,
        construct_ant_timetable,
        update_mmas_pheromones,
    )

    __all__.extend(
        [
            "ACOMemeticSolver",
            "ACOSolver",
            "compute_mmas_bounds",
            "construct_ant_timetable",
            "update_mmas_pheromones",
        ]
    )
except ImportError:  # pragma: no cover
    pass
