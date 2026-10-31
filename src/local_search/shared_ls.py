import numpy as np
from src.core.problem import delta_move
from src.core.instance import Instance

def local_search_first_improvement(inst: Instance, x: np.ndarray, max_iter: int = 100) -> np.ndarray:
    current_x = x.copy()
    n = inst.n_exams
    T = inst.n_slots
    improved = True
    it = 0
    
    while improved and it < max_iter:
        improved = False
        it += 1
        exams = np.arange(n)
        np.random.shuffle(exams)
        
        for i in exams:
            current_slot = current_x[i]
            for new_slot in range(T):
                if new_slot == current_slot:
                    continue
                delta_f, delta_v = delta_move(inst, current_x, i, new_slot)
                if delta_v < 0 or (delta_v == 0 and delta_f < 0):
                    current_x[i] = new_slot
                    improved = True
                    break
            if improved:
                break
    return current_x
