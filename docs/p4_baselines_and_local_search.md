# توثيق المرحلة الرابعة (P4: Baselines, Exact B&B & Shared Local Search)

## 1. ما الذي تم بناؤه في هذه المرحلة؟
توفّر هذه المرحلة أربعة حلّالات مرجعية (Baselines & Exact) متوافقة بالكامل مع الفئة المجردة `Solver` والعقد التقني، بالإضافة إلى حزمة البحث المحلي المشتركة التي تستدعيها الخوارزميات الميميتية في المراحل P5 (GA) و P6 (PSO) و P7 (ACO):

### أ. حزمة البحث المحلي المشتركة (`src/local_search/`)
- `src/local_search/move_ls.py`: مشغّل نقل امتحان منفرد (`move_ls`) بأول تحسين (First-Improvement) أو أفضل تحسين (Best-Improvement)، مع خصم دقيق للميزانية عبر `budget.charge_delta` والتوقف الفوري عند `budget.exhausted`.
- `src/local_search/swap_ls.py`: مشغّل مبادلة فترتي امتحانين (`swap_ls`) مع دالة التقييم التفاضلي الدقيق للمبادلة `evaluate_swap_delta` بتعقيد $O(\deg(i) + \deg(j))$.
- `src/local_search/kempe_ls.py`: مشغّل سلاسل كيمب (`kempe_ls`, `extract_kempe_chain`, `apply_kempe_chain_step`) الذي يبدّل الفترتين $s_1 \leftrightarrow s_2$ عبر المركّبة المتصلة في البيان الجزئي المستحث، مع تقييم سريع للحواف الحدودية للسلسلة فقط (لأن المسافات الزمنية للحواف الداخلية تبقى ثابتة).
- `src/local_search/shared_ls.py`: نقطة الدخول الموحدة `intensify` (تدعم الأنماط `"move"`, `"kempe"`, `"swap"`, `"hybrid"`) والدالة `local_search_first_improvement` لاستخدامها مباشرة في الخوارزميات الهجينة.

### ب. الحلّالات المرجعية الأربعة (`src/baselines/`)
- `src/baselines/random_search.py` (`RandomSearchSolver`, `name="random"`): معاينة عشوائية مستقلة مع إصلاح `repair` وتتبع التنوع والتقارب.
- `src/baselines/dsatur.py` (`DsaturSolver`, `name="dsatur"`): خوارزمية تلوين البيان بدرجة التشبع (Brélaz, 1979) متعددة الانطلاقات (Multi-start) مع اختيار الفترة الأقل كلفة تقارب `min_proximity`.
- `src/baselines/sa.py` (`SimulatedAnnealingSolver`, `name="sa"`): خوارزمية التلدين المحاكى بجوار هجين يجمع بين حركات النقل المفردة وسلاسل كيمب (`kempe_prob=0.25`) مع جدول تبريد هندسي وإعادة تسخين عند الركود (`reheating`).
- `src/baselines/exact_bb.py` (`ExactBranchAndBoundSolver`, `name="exact"`): خوارزمية التفريع والتحديد (Branch & Bound) للمسائل الصغيرة (`Tiny`) مع تهيئة حد علوي ابتدائي (Warm-start) وكسر التناظر والتشذيب التزايدي لكلفة التقارب.
- `configs/{random,dsatur,sa,exact}.yaml`: ملفات الإعداد الرسمية بقسمي `defaults` و `search_space`.
- `tests/test_baselines.py`: اختبارات وحدة شاملة للتحقق من مطابقة العقد، وإعادة الإنتاج بنفس البذرة، وعدم تجاوز الميزانية، وتحسين البحث المحلي.

## 2. كيفية التشغيل والفحص
```bash
# Run baseline and local search unit tests
pytest tests/test_baselines.py -v

# Quick pilot run on tiny-a and small-a
PYTHONPATH=. python3 -c "
from src.data import load_instance
from src.core.solver import Budget
from src.baselines import get_baseline_solver

inst = load_instance('small-a')
for name in ['random', 'dsatur', 'sa']:
    res = get_baseline_solver(name).solve(inst, Budget(1500), seed=0)
    print(name, 'feasible:', res.feasible, 'best_f:', round(res.best_f, 4), 'evals:', round(res.evals, 1))
"
```

## 3. القرارات التصميمية المهمة وأسبابها
1. **تقييم الحواف الحدودية فقط في سلاسل كيمب (`kempe_ls`):**
   عند تبديل الفترتين $s_1$ و $s_2$ لجميع امتحانات المركّبة المتصلة $K$، فإن أي امتحانين $u, v \in K$ ينتقلان معاً وتظل المسافة $|x_u - x_v|$ ثابتة تماماً. لذلك تكتفي دالة `apply_kempe_chain_step` بحساب تغير الكلفة على الحواف الحدودية $(u \in K, v
otin K)$، مما يضمن دقة رياضية تامة وسرعة عالية.
2. **الحفاظ على المقبولية داخل التلدين المحاكى (`sa`):**
   بمجرد وصول `SA` إلى جدول مقبول ($V(x)=0$)، يتم رفض أي حركة مفردة تُحدث تعارضاً صلباً ($\Delta V > 0$)، مع الاعتماد على سلاسل كيمب للقفز بين التلوينات المقبولة المختلفة عندما تكون القيود ضيقة.
