# العقد التقني المشترك (Interface Contract)

هذا الملف يوثق العقد التقني المجمّد بين مراحل المشروع (إصدار `v0.0-contract`) بالتنسيق بين P1 و P3. أي تعديل على التواقيع أو هياكل البيانات بعد التجميد يتطلب فتح Issue بوسم `contract-change` وموافقة P1 و P3 وإشعار باقي الفريق.

## 3.1 الصياغة الرياضية

نرقّم الامتحانات من $0$ إلى $N-1$، والفترات الزمنية من $0$ إلى $T-1$، وعدد الطلاب الكلي $S$. نعرّف $c_{ij}$ بأنه عدد الطلاب المسجلين في الامتحانين $i$ و $j$ معاً.

متجه القرار هو متجه أعداد صحيحة $x$ بطول $N$:
$$x \in \{0, \dots, T-1\}^N$$

القيد الصلب (منع وجود امتحانين لنفس الطالب في نفس الفترة):
$$c_{ij} > 0 \implies x_i \neq x_j$$

أوزان التقارب بحسب المسافة بين الفترتين $d = |x_i - x_j|$:
$$
w(d)=\begin{cases}
0 & d=0\\
2^{5-d} & d=1,\ldots,5\\
0 & d>5
\end{cases}
$$

يُستخدم الاصطلاح \(w(0)=0\) عند تقييم جدول غير مقبول أثناء البحث حتى لا تُضاف كلفة تقارب للأزواج الموضوعة في الفترة نفسها؛ وتُسجّل مخالفاتها في \(V(x)\).

دالة الهدف الناعم (Carter proximity cost):
$$f(x) = \frac{1}{S} \sum_{i < j} c_{ij} \cdot w(|x_i - x_j|)$$

وزن الانتهاكات ودالة اللياقة مع العقوبة:
$$V(x) = \sum_{i < j,\; x_i = x_j} c_{ij} \qquad \Big| \qquad F(x) = f(x) + \lambda \cdot \frac{V(x)}{S} \quad (\lambda = 1000.0)$$

ملاحظات أساسية للتطبيق والتقرير:
1. التقرير والمقارنات النهائية تستخدم $f(x)$ للحل الأفضل المقبول (feasible) فقط. إذا لم تجد الخوارزمية حلاً مقبولاً في تشغيل معين، يُسجّل `best_f = float("inf")` وتُحسب نسبة التشغيلات المقبولة (feasibility rate) كمقياس مستقل.
2. الدالة $F(x)$ تُستخدم داخلياً لتوجيه البحث عند السماح بحلول غير مقبولة (نسخة penalty). النسخة الافتراضية تعتمد على الإصلاح (`repair`)، والمقارنة بين الاثنين (ablation) من مسؤولية P5 و P8.
3. عدد الفترات $T$ ثابت لكل instance، ولا توجد قيود على سعة القاعات في النسخة الأساسية، والمسافة بين الفترات هي الفرق في الفهرس $|x_i - x_j|$.

## 3.2 هياكل البيانات والتواقيع

```python
# src/core/instance.py
@dataclass(frozen=True)
class Instance:
    name: str
    n_exams: int
    n_students: int
    n_slots: int
    indptr: np.ndarray     # int32, len N+1 (conflict graph in CSR form)
    indices: np.ndarray    # int32, neighbours j of each exam i
    weights: np.ndarray    # int32, c_ij aligned with `indices`
    exam_size: np.ndarray  # int32, enrolment per exam
    n_edges: int           # |E| = number of unordered pairs with c_ij > 0
    meta: dict             # source, generator params, seed, planted solution (optional)

# src/core/problem.py (target design: Numba kernels with thin Python wrappers)
proximity_cost(inst, x) -> float                            # f(x)
violation_weight(inst, x) -> float                          # V(x)
fitness(inst, x, lam=1000.0) -> float                       # F(x)
delta_move(inst, x, i, s_new) -> tuple[float, float]        # (delta_f, delta_V), O(deg(i))
is_feasible(inst, x) -> bool
repair(inst, x, rng, max_passes=5) -> np.ndarray
random_solution(inst, rng) -> np.ndarray                    # random + repair
diversity(X: np.ndarray) -> float                           # mean normalized pairwise Hamming distance

# src/core/solver.py
class Budget:
    def __init__(self, max_evals: float, max_seconds: float | None = None): ...
    def charge_full(self, n: int = 1) -> None: ...
    def charge_delta(self, inst: Instance, i: int) -> None: ...  # adds deg(i)/|E|
    exhausted: bool (property)
    used: float (property)

@dataclass
class Result:
    best_x: np.ndarray
    best_f: float
    feasible: bool
    history: np.ndarray  # rows: [evals_used, iteration, best_f_so_far, seconds, diversity]
    elapsed: float
    evals: float
    hparams: dict

class Solver(ABC):
    name: str
    def __init__(self, **hparams): ...
    @abstractmethod
    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result: ...
```

## 3.3 قاعدة الميزانية العادلة (Fair Budget Rule)

نقارن الخوارزميات بنفس ميزانية التقييمات وليس بنفس عدد الـ iterations، ووحدة الميزانية هي مكافئ تقييم كامل (Full Evaluation Equivalent - FEE):

| العملية | الكلفة بوحدة FEE | السبب |
| :--- | :--- | :--- |
| تقييم كامل لحل (`fitness`) | `1.0` | التعريف الأساسي |
| تقييم حركة واحدة (`delta_move`) للامتحان $i$ | `deg(i) / |E|` | كلفتها متناسبة مع عدد جيران $i$ فهي جزء من تقييم كامل |
| فرد في GA، جسيم في PSO بعد فك الترميز، نملة في ACO بعد اكتمال بنائها | `1.0` | كل حل كامل يُقيّم مرة واحدة |
| حسابات الـ heuristic داخل بناء ACO، وفك الترميز الجشع في PSO | لا تُحتسب (`0.0`) | تظهر في زمن التنفيذ (wall-clock) وتُناقش في التقرير |

- الميزانيات المبدئية: `Tiny = 20,000` و`Small = 50,000` و`Medium = 200,000` و`Large = 500,000` FEE.
- عند \(|E|=0\)، لا تُستخدم الصيغة \(\deg(i)/|E|\) حرفياً؛ يتعامل التنفيذ مع هذه الحالة بصورة خاصة لتجنب القسمة على الصفر.
- تتوقف كل خوارزمية فور تحقق `budget.exhausted` حتى لو كانت في منتصف الـ iteration، والحل المُسلّم هو أفضل حل مقبول وُجد قبل نفاد الميزانية.

## 3.4 السجلات، البذور، المجموعات، والأسماء

- **Tracker:** يسجل أفضل قيمة مقبولة كل 0.5% من الميزانية (نحو 200 نقطة) مع رقم الـ iteration والزمن والتنوع (`diversity`).
- **البذور (seeds):** التجارب النهائية من `0` إلى `29` (30 تشغيلاً). ضبط الـ hyperparameters من `1000` إلى `1004`. بذور توليد الـ instances مثبتة في `data/instances.yaml`.
- **أسماء الحلّالات:** `random`، `dsatur`، `sa`، `exact`، `ga`، `ga_memetic`، `pso`، `pso_memetic`، `aco`، `aco_memetic`. لكل منها ملف `configs/<name>.yaml` بقسمين: `defaults` و `search_space`.
