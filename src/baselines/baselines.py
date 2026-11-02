import numpy as np
from typing import Tuple
from src.core.instance import Instance
from src.core.problem import random_solution, fitness, is_feasible, delta_move, proximity_cost

def random_search(inst: Instance, budget, seed: int) -> Tuple[np.ndarray, float, bool]:
    rng = np.random.default_rng(seed)
    best_x = random_solution(inst, rng)
    budget.charge_full(1)
    best_f = fitness(inst, best_x)
    best_feasible_x = best_x.copy() if is_feasible(inst, best_x) else None
    best_feasible_f = float('inf')
    if best_feasible_x is not None:
        best_feasible_f = fitness(inst, best_feasible_x)

    while not budget.exhausted:
        x = random_solution(inst, rng)
        budget.charge_full(1)
        f = fitness(inst, x)
        if f < best_f:
            best_f = f
            best_x = x.copy()
        if is_feasible(inst, x) and f < best_feasible_f:
            best_feasible_f = f
            best_feasible_x = x.copy()

    return (best_feasible_x if best_feasible_x is not None else best_x, 
            best_feasible_f if best_feasible_x is not None else best_f, 
            best_feasible_x is not None)

def dsatur_heuristic(inst: Instance, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = inst.n_exams
    T = inst.n_slots
    x = -1 * np.ones(n, dtype=np.int32)
    degrees = np.array([len(inst.indices[inst.indptr[i]:inst.indptr[i+1]]) for i in range(n)])
    unassigned = set(range(n))
    
    while unassigned:
        best_exam = max(unassigned, key=lambda e: (
            len({x[nbr] for nbr in range(inst.indptr[e], inst.indptr[e+1]) if x[inst.indices[nbr]] != -1}),
            degrees[e]
        ))
        forbidden_slots = {x[inst.indices[nbr]] for nbr in range(inst.indptr[best_exam], inst.indptr[best_exam+1]) if x[inst.indices[nbr]] != -1}
        available_slots = [s for s in range(T) if s not in forbidden_slots]
        
        if available_slots:
            x[best_exam] = rng.choice(available_slots)
        else:
            x[best_exam] = rng.integers(0, T)
        unassigned.remove(best_exam)
        
    return x
