import time
import numpy as np
from src.core.instance import Instance
from src.core.solver import Solver, Result, Budget
from src.core.problem import fitness, random_solution, is_feasible, diversity, repair, delta_move

class GASolver(Solver):
    """Pure Genetic Algorithm solver implementation."""
    name = "ga"
    
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.pop_size = self.hparams.get("pop_size", 50)
        self.mutation_rate = self.hparams.get("mutation_rate", 0.1)
        self.tournament_size = self.hparams.get("tournament_size", 3)
        self.crossover_rate = self.hparams.get("crossover_rate", 0.8)

    def _select_parents(self, scored_population, rng):
        """Select parents using tournament selection."""
        selected = []
        pop_size = len(scored_population)
        for _ in range(pop_size):
            aspirants = [scored_population[rng.integers(0, pop_size)] for _ in range(self.tournament_size)]
            best_aspirant = min(aspirants, key=lambda x: x[0])
            selected.append(best_aspirant[1])
        return selected

    def _crossover_and_mutate(self, population, inst, rng):
        """Generate new population through crossover mutation and repair"""
        new_population = []
        pop_size = len(population)
        
        for i in range(0, pop_size, 2):
            p1 = population[i]
            p2 = population[(i + 1) % pop_size]
            
            if rng.random() < self.crossover_rate:
                mask = rng.random(size=p1.shape) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = np.copy(p1), np.copy(p2)
                
            for j in range(len(c1)):
                if rng.random() < self.mutation_rate:
                    c1[j] = rng.integers(0, inst.n_slots)
            c1 = repair(inst, c1, rng)
            
            for j in range(len(c2)):
                if rng.random() < self.mutation_rate:
                    c2[j] = rng.integers(0, inst.n_slots)
            c2 = repair(inst, c2, rng)
            
            new_population.extend([c1, c2])
            
        return new_population[:pop_size]

    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        start_time = time.time()
        rng = np.random.default_rng(seed)

        population = [random_solution(inst, rng) for _ in range(self.pop_size)]

        best_x = None
        best_f = float('inf')
        feasible_found = False
        history_list = []
        iteration = 0

        while not budget.exhausted:
            scored_population = []
            X_matrix = np.array(population)

            for ind in population:
                if budget.exhausted:
                    break
                
                score = fitness(inst, ind)
                budget.charge_full(1.0)
                scored_population.append((score, ind))

                if is_feasible(inst, ind) and score < best_f:
                    best_f = score
                    best_x = np.copy(ind)
                    feasible_found = True

            if not scored_population:
                break

            scored_population.sort(key=lambda x: x[0])
            
            div_val = diversity(X_matrix)
            elapsed = time.time() - start_time
            history_list.append([budget.used, iteration, best_f, elapsed, div_val])

            best_current_ind = scored_population[0][1]
            parents = self._select_parents(scored_population, rng)
            population = self._crossover_and_mutate(parents, inst, rng)
            
            # Elitism injection
            population[0] = np.copy(best_current_ind)

            iteration += 1

        elapsed_total = time.time() - start_time
        history_arr = np.array(history_list) if history_list else np.empty((0, 5))

        return Result(
            best_x=best_x if best_x is not None else population[0],
            best_f=best_f if feasible_found else float('inf'),
            feasible=feasible_found,
            history=history_arr,
            elapsed=elapsed_total,
            evals=budget.used,
            hparams=self.hparams
        )


class MemeticGASolver(GASolver):
    """Memetic Genetic Algorithm solver adding local search via delta move"""
    name = "ga_memetic"

    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.ls_freq = self.hparams.get("ls_freq", 5) # Apply local search every X iterations

    def _local_search(self, x, inst, budget, rng):
        """Refine individual solution using delta evaluations."""
        lam = 1000.0
        S = inst.n_students
        n_exams = inst.n_exams
        
        # Select a subset of exams to optimize locally to respect budget
        sample_size = max(1, n_exams // 10)
        exams_to_check = rng.choice(n_exams, size=sample_size, replace=False)

        for i in exams_to_check:
            if budget.exhausted:
                break
            current_slot = x[i]
            new_slot = rng.integers(0, inst.n_slots)
            if new_slot == current_slot:
                continue

            delta_f, delta_V = delta_move(inst, x, i, new_slot)
            delta_F = delta_f + lam * (delta_V / S)
            
            # Charge delta move cost according to the Fair Budget Rule
            budget.charge_delta(inst, i)

            if delta_F < 0:
                x[i] = new_slot
                
        return repair(inst, x, rng)

    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        # Override solve to inject local search periodically on elite solutions
        start_time = time.time()
        rng = np.random.default_rng(seed)

        population = [random_solution(inst, rng) for _ in range(self.pop_size)]

        best_x = None
        best_f = float('inf')
        feasible_found = False
        history_list = []
        iteration = 0

        while not budget.exhausted:
            scored_population = []
            X_matrix = np.array(population)

            for ind in population:
                if budget.exhausted:
                    break
                
                score = fitness(inst, ind)
                budget.charge_full(1.0)
                scored_population.append((score, ind))

                if is_feasible(inst, ind) and score < best_f:
                    best_f = score
                    best_x = np.copy(ind)
                    feasible_found = True

            if not scored_population:
                break

            scored_population.sort(key=lambda x: x[0])
            
            div_val = diversity(X_matrix)
            elapsed = time.time() - start_time
            history_list.append([budget.used, iteration, best_f, elapsed, div_val])

            best_current_ind = scored_population[0][1]

            # Apply Memetic Local Search periodically
            if iteration % self.ls_freq == 0 and not budget.exhausted:
                best_current_ind = self._local_search(best_current_ind, inst, budget, rng)

            parents = self._select_parents(scored_population, rng)
            population = self._crossover_and_mutate(parents, inst, rng)
            
            population[0] = np.copy(best_current_ind)
            iteration += 1

        elapsed_total = time.time() - start_time
        history_arr = np.array(history_list) if history_list else np.empty((0, 5))

        return Result(
            best_x=best_x if best_x is not None else population[0],
            best_f=best_f if feasible_found else float('inf'),
            feasible=feasible_found,
            history=history_arr,
            elapsed=elapsed_total,
            evals=budget.used,
            hparams=self.hparams
        )