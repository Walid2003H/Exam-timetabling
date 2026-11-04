# توثيق المرحلة السابعة (P7: Ant Colony Optimization — `aco` & `aco_memetic`)

## 1. ما الذي تم بناؤه في هذه المرحلة؟
تنفّذ هذه المرحلة خوارزمية مستعمرة النمل بنظام الحدين الأعلى والأدنى ($\mathcal{MAX}	ext{-}\mathcal{MIN}$ Ant System - MMAS) استناداً إلى عمل Stützle & Hoos (2000) وتكييف Socha, Knowles, & Samples (2002) لمسائل الجدولة الجامعية، وذلك عبر النسختين الرسميتين المعتمدتين في العقد التقني (`docs/interface_contract.md`):
- `src/algorithms/aco.py`:
  - **تمثيل الفيرومون وحدود MMAS (`compute_mmas_bounds`, `update_mmas_pheromones`):** مصفوفة فيرومون $	au \in \mathbb{R}^{N 	imes T}$ يمثل العنصر $	au_{i,s}$ فيها الخبرة التراكمية لإسناد الامتحان $i$ إلى الفترة الزمنية $s$. بعد كل جيل، يتبخر الفيرومون بمعامل $
ho$ ويُودع أثر جديد من أفضل نملة في الجيل (`ibest`) بالتناوب مع أفضل حل عالمي (`gbest`)، ثم تُقيد جميع القيم ضمن المجال الديناميكي $[	au_{\min}, 	au_{\max}]$ مع تنعيم المسارات (Trail Smoothing) عند الركود.
  - **آلية بناء النملة (`construct_ant_timetable`):** ترتّب كل نملة الامتحانات ديناميكياً وفق درجة التشبع والدرجة الساكنة (أسلوب DSATUR)، ثم تختار الفترة الزمنية للامتحان $i$ من بين الفترات الأقل تعارضاً بالاعتماد على جداء الفيرومون والمعلومة الإرشادية $	au_{i,s}^lpha \cdot \eta_{i,s}^eta$ مع احتمال استغلال جشع $q_0$. وطبقاً للقسم 3.3 من العقد التقني، لا تُحتسب حسابات الـ heuristic الداخلية في ميزانية FEE بل يُخصم $1.0	ext{ FEE}$ عند تقييم الجدول المكتمل للنملة.
  - **الفئتان الرسميتان:** `ACOSolver` (`name="aco"`) للنسخة النقية، و `ACOMemeticSolver` (`name="aco_memetic"`) للنسخة الميميتية التي تدمج بناء المستعمرة مع وحدة البحث المحلي المشتركة `intensify` (`move_ls + kempe_ls + swap_ls`) من `src/local_search/shared_ls.py`.
- `configs/aco.yaml` و `configs/aco_memetic.yaml`: الإعدادات الافتراضية (`defaults`) وفضاء البحث (`search_space`) لعدد النمل `n_ants`، والمعاملات $lpha, eta, 
ho, q_0$، ومعاملات البحث المحلي.
- `tests/test_aco.py`: اختبارات وحدة شاملة للتحقق من تقيد الفيرومون ضمن $[	au_{\min}, 	au_{\max}]$، وبناء الجداول، والنسختين `aco` و `aco_memetic`، ووضعي `repair` و `penalty`، وإعادة الإنتاج بنفس البذرة.

## 2. كيفية التشغيل والفحص
```bash
# Run ACO unit tests
pytest tests/test_aco.py -v

# Quick pilot run on small-a and medium-a
PYTHONPATH=. python3 -c "
from src.data import load_instance
from src.core.solver import Budget
from src.algorithms.aco import ACOSolver, ACOMemeticSolver

inst = load_instance('small-a')
for solver in [ACOSolver(), ACOMemeticSolver()]:
    res = solver.solve(inst, Budget(1500), seed=0)
    print(solver.name, 'best_f:', round(res.best_f, 4), 'feasible:', res.feasible)
"
```

## 3. القرارات التصميمية المهمة وأسبابها
1. **إقران الفيرومون على الزوج (امتحان، فترة) $	au_{i,s}$ بدلاً من الأزواج (امتحان، امتحان):**
   في مسألة جدولة الامتحانات ذات عدد الفترات الثابت $T$، يتطلب تمثيل الفيرومون المباشر بين الامتحانات والفترات مصفوفة بحجم $N 	imes T$ فقط (نحو $500 	imes 35$ في المسائل الكبيرة)، مما يجعل تحديث الفيرومون واسترجاعه سريعاً جداً وموجهاً مباشرة نحو الفترات الزمنية الواعدة.
2. **التناوب بين أفضل نملة في الجيل (`ibest`) وأفضل نملة شاملة (`gbest`):**
   الاعتماد الحصري على `gbest` منذ البداية يؤدي إلى تركيز الفيرومون مبكراً حول أول جدول مقبول، بينما يمنح التناوب (`gbest_deposit_freq = 3`) مع حصر القيم في $[	au_{\min}, 	au_{\max}]$ توازناً عملياً بين الاستكشاف والاستغلال.
