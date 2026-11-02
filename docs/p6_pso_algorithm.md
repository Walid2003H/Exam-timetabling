# توثيق المرحلة السادسة (P6: Particle Swarm Optimization — `pso` & `pso_memetic`)

## 1. ما الذي تم بناؤه في هذه المرحلة؟
تنفّذ هذه المرحلة خوارزمية تحسين سرب الجسيمات (Particle Swarm Optimization - PSO) بنسختيها النقية (`pso`) والميميتية الهجينة (`pso_memetic`) وبترميزين توافقيين مختلفين وفق متطلبات العقد التقني (`docs/interface_contract.md`):
- `src/algorithms/pso.py`:
  - **الترميز الأول — المفاتيح العشوائية (`encoding="random_keys"`):** استناداً إلى Bean (1994) و Kennedy & Eberhart (1995)، يتحرك كل جسيم في الفضاء المستمر $[0, 1]^N$ عبر تحديث السرعة والموقع، ثم يفكّ الترميز (`decode_random_keys`) إما بأسلوب الأولوية الجشعة (`priority_greedy`) الذي يدمج المفتاح المستمر مع الدرجة الساكنة للامتحان ويسنده إلى الفترة الأقل تعارضاً وكلفة تقارب، وإما بأسلوب الإسقاط المباشر (`slot_map`) لأغراض دراسة الاستئصال (`repair` مقابل `penalty`).
  - **الترميز الثاني — السرب المتقطع المباشر (`encoding="direct_discrete"`):** يتحرك الجسيم مباشرة في فضاء الفترات $\{0, \dots, T-1\}^N$ عبر متجه احتمالات سرعة `vel_prob` محكوم بمعاملات العطالة $w$ والمعرفة الشخصية $c_1$ والاجتماعية $c_2$، وينسخ الفترات من أفضل موقع شخصي (`pbest`) أو أفضل موقع عالمي (`gbest`) مع طفرة استكشافية طفيفة (`mutation_rate`).
  - **الفئتان الرسميتان:** `PSOSolver` (`name="pso"`) و `PSOMemeticSolver` (`name="pso_memetic"` التي تستدعي وحدة البحث المحلي المشتركة `intensify` من `src/local_search/shared_ls.py` مع إعادة ترميز لاماركية Lamarckian write-back إلى موقع الجسيم).
- `configs/pso.yaml` و `configs/pso_memetic.yaml`: الإعدادات الافتراضية (`defaults`) وفضاء البحث (`search_space`) لضبط المعاملات الفائقة في المرحلة الثامنة (P8).
- `tests/test_pso.py`: اختبارات وحدة شاملة للتحقق من الترميزين، والنسختين النقية والميميتية، ووضعي `repair` و `penalty`، وإعادة الإنتاج بنفس البذرة، والالتزام الصارم بميزانية FEE.

## 2. كيفية التشغيل والفحص
```bash
# Run PSO unit tests
pytest tests/test_pso.py -v

# Quick pilot comparison on small-a
PYTHONPATH=. python3 -c "
from src.data import load_instance
from src.core.solver import Budget
from src.algorithms.pso import PSOSolver, PSOMemeticSolver

inst = load_instance('small-a')
for solver in [
    PSOSolver(encoding='random_keys'),
    PSOSolver(encoding='direct_discrete'),
    PSOMemeticSolver(encoding='random_keys'),
]:
    res = solver.solve(inst, Budget(1500), seed=0)
    print(solver.name, solver.hparams['encoding'], 'best_f:', round(res.best_f, 4), 'feasible:', res.feasible)
"
```

## 3. القرارات التصميمية المهمة وأسبابها
1. **فك الترميز الموجه بالأولوية (`priority_greedy`) في Random-Keys:**
   الإسقاط الكمي البسيط $\lfloor z_i \cdot T 
floor$ يولّد أعداداً كبيرة من التعارضات الصلبة لأن الجسيمات لا تدرك بنية بيان التعارض أثناء تحركها في $[0,1]^N$. أما استخدام المفاتيح المستمرة لتعديل أولوية ترتيب الامتحانات ثم إسناد كل امتحان إلى الفترة الخالية من التعارض الأقل كلفة تقارب (مع عدم احتساب كلفة فك الترميز في FEE طبقاً للقسم 3.3 من العقد التقني)، فيمنح السرب قدرة عالية على استكشاف تلوينات مقبولة ومنخفضة الكلفة.
2. **الكتابة العكسية اللاماركية (`encode_slots_to_keys`) بعد الإصلاح والبحث المحلي:**
   عندما يقوم مشغّل `repair` أو البحث المحلي `intensify` بتحسين جدول الجسيم $x_k$، تُحدَّث مفاتيح `pbest_keys[k]` لتوافق الجدول المحسّن حتى ينجذب السرب في الأجيال التالية نحو المنطقة المحسّنة فعلياً.
