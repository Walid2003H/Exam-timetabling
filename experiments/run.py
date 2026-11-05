import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import json
import csv
import time
from pathlib import Path
import numpy as np

from src.core.solver import Budget, Solver, Result
from src.core.instance import Instance
from src.core.problem import proximity_cost, violation_weight

# ==============================================================================
# استيراد جميع الخوارزميات الحقيقية من ملفات الفريق
# ==============================================================================
from src.baselines.random_search import RandomSearchSolver
from src.baselines.dsatur import DsaturSolver
from src.baselines.exact_bb import ExactBranchAndBoundSolver
from src.baselines.sa import SimulatedAnnealingSolver
from src.algorithms.ga import GASolver, GAMemeticSolver
from src.algorithms.pso import PSOSolver, PSOMemeticSolver
from src.algorithms.aco import ACOSolver, ACOMemeticSolver

# ==============================================================================
# دالة تحميل البيانات الحقيقية من ملفات .npz
# ==============================================================================
def load_instance_from_npz(path: str) -> Instance:
    """تحميل البيانات الحقيقية من ملفات .npz"""
    data = np.load(path)
    
    return Instance(
        name=Path(path).stem,
        n_exams=int(data['n_exams']),
        n_students=int(data['n_students']),
        n_slots=int(data['n_slots']),
        indptr=data['indptr'],
        indices=data['indices'],
        weights=data['weights'],
        exam_size=data['exam_size'],
        n_edges=int(data['n_edges']),
        meta={}
    )

# ==============================================================================
# دالة مصنع الخوارزميات (Solver Factory)
# ==============================================================================
def get_solver(algo_name: str):
    """تقوم بإرجاع الكلاس المناسب للخوارزمية بناءً على اسمها"""
    algorithms = {
        "random": RandomSearchSolver,
        "dsatur": DsaturSolver,
        "exact": ExactBranchAndBoundSolver,
        "sa": SimulatedAnnealingSolver,
        "ga": GASolver,
        "ga_memetic": GAMemeticSolver,
        "pso": PSOSolver,
        "pso_memetic": PSOMemeticSolver,
        "aco": ACOSolver,
        "aco_memetic": ACOMemeticSolver
    }
    
    if algo_name not in algorithms:
        raise ValueError(f"Algorithm '{algo_name}' is not implemented or imported.")
    
    return algorithms[algo_name]()

# ==============================================================================
# الحلقة الرئيسية للتشغيل
# ==============================================================================
def main():
    print("Starting P8 Experiments Runner...")
    
    with open('experiments/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    results = []
    
    for algo_name in config['algorithms']:
        for inst_cfg in config['instances']:
            inst_name = inst_cfg['name']
            inst_path = inst_cfg['path']
            
            if not Path(inst_path).exists():
                print(f"[Error] Instance file not found: {inst_path}. Skipping...")
                continue

            # استخدام الدالة الجديدة لتحميل ملفات npz
            inst = load_instance_from_npz(inst_path)
            
            for seed in config['seeds']:
                # 1. تجهيز الميزانية
                budget = Budget(
                    max_evals=config['budget']['max_evals'],
                    max_seconds=config['budget']['max_seconds']
                )
                
                # 2. استدعاء الخوارزمية
                try:
                    solver = get_solver(algo_name)
                except ValueError as e:
                    print(e)
                    continue
                
                print(f"Running {algo_name} on {inst_name} (Seed: {seed})...")
                result = solver.solve(inst, budget, seed)
                
                # 3. التحقق النهائي من الصلاحية
                final_feasible = result.feasible
                if final_feasible:
                    v_check = violation_weight(inst, result.best_x)
                    if v_check > 1e-9:
                        final_feasible = False
                        print(f"  [Warning] Solver claimed feasible but V(x)={v_check}")

                # 4. حفظ النتائج
                row = {
                    'algorithm': algo_name,
                    'instance': inst_name,
                    'seed': seed,
                    'best_f': result.best_f,
                    'feasible': final_feasible,
                    'evals_used': result.evals,
                    'elapsed': result.elapsed,
                    'hparams': json.dumps(result.hparams)
                }
                results.append(row)

      # 5. حفظ النتائج في مجلد results/raw باسم ديناميكي
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    n_seeds = len(config['seeds'])
    budget_val = config['budget']['max_evals']
    filename = f"run_{timestamp}_{int(budget_val)}fee_{n_seeds}seeds.csv"
    
    out_dir = Path('results/raw')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / filename
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    print(f"\nSuccess! Results saved to {out_file}")

if __name__ == "__main__":
    main()