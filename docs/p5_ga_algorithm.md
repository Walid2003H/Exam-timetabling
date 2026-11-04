# توثيق المرحلة الخامسة (P5: Genetic Algorithm — `ga` & `ga_memetic`)

## 1. ما الذي تم بناؤه في هذه المرحلة؟
تنفّذ هذه المرحلة الخوارزمية الجينية (Genetic Algorithm) بنسختيها النقية (`ga`) والميميتية الهجينة (`ga_memetic`) وفق إطار Burke, Newall, & Weare (1996) وبالتزام كامل بالعقد التقني المشترك (`docs/interface_contract.md`):
- `src/algorithms/ga.py`:
  - **تمثيل الكروموسوم وتهيئة المجتمع:** يمثَّل كل فرد بمتجه الفترات المباشر $x \in \{0, \dots, T-1\}^N$ من النوع `int32`. يُهيَّأ المجتمع الابتدائي بمزيج متوازن (35% عبر البناء الموجه بـ `dsatur_construct` و 65% عبر التوليد العشوائي المُصلَح) للجمع بين جودة الانطلاق والتنوع الجيني.
  - **الانتخاب والتصالب والطفرة (`tournament_select`, `crossover_chromosomes`, `mutate_chromosome`):** انتخاب البطولة $k$-way Tournament Selection، مع خيارين للتصالب (`uniform` الموحد و `two_point` ذي النقطتين)، وطفرة إعادة إسناد الفترة الزمنية بمعدل `mutation_rate` مع الاحتفاظ بالنخبة (`elitism_count = 2`).
  - **آلية التعامل مع القيود (`repair` مقابل `penalty`):** تدعم الفئة `GASolver` كلاً من وضع الإصلاح الافتراضي (`constraint_handling="repair"`) ووضع العقوبة (`constraint_handling="penalty"`) لأغراض دراسة الاستئصال المشتركة مع المرحلة الثامنة (P8).
  - **الفئتان الرسميتان:** `GASolver` (`name="ga"`) للنسخة الجينية النقية، و `GAMemeticSolver` (مع الاسم المرادف `MemeticGASolver` للتوافق، `name="ga_memetic"`) التي تدمج التطور الجيني مع وحدة البحث المحلي المشتركة `intensify` (`move_ls + kempe_ls + swap_ls`) من `src/local_search/shared_ls.py`.
- `configs/ga.yaml` و `configs/ga_memetic.yaml`: الإعدادات الافتراضية (`defaults`) وفضاء البحث (`search_space`) لحجم المجتمع `pop_size`، وحجم البطولة `tournament_size`، ومعدل ونوع التصالب، ومعدل الطفرة، ومعاملات البحث المحلي.
- `tests/test_ga.py`: اختبارات وحدة شاملة لفحص مشغلات الانتخاب والتصالب والطفرة، والنسختين `ga` و `ga_memetic`، ووضعي `repair` و `penalty`، وإعادة الإنتاج الحتمية بنفس البذرة.

## 2. كيفية التشغيل والفحص
```bash
# Run GA unit tests
pytest tests/test_ga.py -v

# Quick pilot comparison on small-a
PYTHONPATH=. python3 -c "
from src.data import load_instance
from src.core.solver import Budget
from src.algorithms.ga import GASolver, GAMemeticSolver

inst = load_instance('small-a')
for solver in [GASolver(), GAMemeticSolver()]:
    res = solver.solve(inst, Budget(1500), seed=0)
    print(solver.name, 'best_f:', round(res.best_f, 4), 'feasible:', res.feasible)
"
```

## 3. القرارات التصميمية المهمة وأسبابها
1. **إدماج الإصلاح السريع بعد الطفرة (`constraint_handling="repair"`):**
   مشغلات التصالب القياسية (مثل `uniform` و `two_point`) تنقل فترات الامتحانات من الأبوين بشكل مستقل، مما قد يورّث تعارضات صلبة في الذرية. تطبيق مشغّل `repair` التزايدي يعيد الذرية فوراً إلى الفضاء المقبول ($V(x)=0$) قبل التقييم الكامل، بينما يُتاح وضع `penalty` بدون إصلاح للمقارنة التجريبية في دراسة الاستئصال.
2. **التكثيف الميميتي المرحلي وحقن المهاجرين عند الركود:**
   تطبّق `GAMemeticSolver` البحث المحلي المشترك `intensify` عند منتصف الميزانية وفي مرحلة الصقل الختامي مع عزل مولد العشوائية `ls_rng`، مما يمنع استنزاف ميزانية الأجيال المبكرة ويضمن تفوق النسخة الميميتية على النسخة الجينية النقية.
