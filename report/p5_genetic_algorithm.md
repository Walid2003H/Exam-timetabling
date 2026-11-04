# القسم الخامس من التقرير: الخوارزمية الجينية (GA) والخوارزمية الميميتية الهجينة

## 1. تصميم الخوارزمية الجينية لجدولة الامتحانات (`src/algorithms/ga.py`)

تعتمد الخوارزمية الجينية (Genetic Algorithm - GA) على تطوير مجتمع مكوّن من $P$ جدول امتحانات مرشح $X = \{x^{(1)}, \dots, x^{(P)}\}$ عبر أجيال متعاقبة من الانتخاب والتصالب والطفرة. واستناداً إلى أعمال Burke, Newall, & Weare (1996) في الجدولة الآلية للامتحانات، صُممت مكونات `GASolver` (`name="ga"`) على النحو الآتي:

### 1.1 التمثيل الجيني وتهيئة المجتمع الهجينة
يُمثَّل كل كروموسوم مباشرة بمتجه الفترات الصحيح $x \in \{0, 1, \dots, T-1\}^N$ (`dtype=int32`). ولتفادي بطء الانطلاق من جداول عشوائية بحتة دون التضحية بالتنوع الجيني، يُولَّد $35\%$ من أفراد الجيل الابتدائي عبر البناء الإرشادي `dsatur_construct` ببذور فرعية مختلفة، في حين يُولَّد الـ $65\%$ الباقون عشوائياً مع تطبيق مشغّل الإصلاح `repair`. ويُحتسب تقييم كل فرد مكتمل بمقدار $1.0	ext{ FEE}$ وفق قاعدة الميزانية العادلة.

### 1.2 المشغلات التطورية: الانتخاب، التصالب، والطفرة
1. **انتخاب البطولة ($k$-Tournament Selection):** لاختيار كل أب، يُسحب $k = 3$ أفراد عشوائياً من المجتمع الحالي ويُختار الفرد ذو اللياقة المعممة الأدنى $F(x) = f(x) + \lambda V(x)/S$.
2. **التصالب (`crossover_chromosomes`):** باحتمال $p_c = 0.85$، يتزاوج الأبوان $p^{(1)}, p^{(2)}$ لإنتاج فردين جديدين $c^{(1)}, c^{(2)}$ عبر أحد مشغلين:
   - *التصالب الموحد (`uniform`):* يرث الجين $i$ في $c^{(1)}$ قيمته من $p^{(1)}$ أو $p^{(2)}$ باحتمال متساوٍ $0.5$، ويأخذ $c^{(2)}$ القيمة المتممة.
   - *التصالب ذو النقطتين (`two_point`):* يتم اختيار نقطتي قطع $0 \le a < b < N$ وتبديل المقطع الأوسط $[a, b)$ بين الأبوين.
3. **الطفرة والنخبوية (`mutate_chromosome` & Elitism):** يتعرض كل امتحان في الذرية باحتمال $p_m = 0.04$ لإعادة إسناد عشوائية إلى فترة جديدة من $\{0, \dots, T-1\}$، كما يُحفظ أفضل فردين (`elitism_count = 2`) من الجيل السابق مباشرة في الجيل الجديد دون إعادة تقييمهما. وعند ركود المجتمع لـ $14$ جيلاً متتالياً، يُستبدل الربع الأسوأ من المجتمع بأفراد مهاجرين ناتجين عن طفرة موسّعة لأفضل حل مكتشف.

### 1.3 معالجة القيود الصلبة: الإصلاح (`repair`) مقابل العقوبة (`penalty`)
يدعم الكود طريقتين لمعالجة التعارضات الصلبة الناتجة عن التصالب والطفرة:
- **نسخة الإصلاح الافتراضية (`constraint_handling="repair"`):** يُمرَّر كل فرد مولّد عبر مشغّل `repair` لإزالة التعارضات الصلبة قبل حساب لياقته، مما يحصر التطور داخل منطقة الجداول المقبولة ($V(x)=0$).
- **نسخة العقوبة (`constraint_handling="penalty"`):** يُقيَّم الفرد مباشرة بدالة العقوبة $F(x) = f(x) + 1000 \cdot V(x)/S$ دون إصلاح، وتُستخدم هذه النسخة في دراسة الاستئصال (Ablation Study) لقياس أثر الإصلاح الجشع على نسبة المقبولية وجودة الحلول.

## 2. الخوارزمية الجينية الميميتية (`ga_memetic`)

تدمج الفئة `GAMemeticSolver` (`name="ga_memetic"`) البحث التطوري السكاني مع وحدة البحث المحلي المشتركة `intensify` (`move_ls + kempe_ls + swap_ls`):
- عند منتصف الميزانية، يُطبَّق البحث المحلي الهجين على نخبة الجيل ويُحقن الجدول المحسّن في المجتمع لتوجيه التصالب في الأجيال اللاحقة.
- في المرحلة الختامية من الميزانية ($6\%$)، يُنفَّذ صقل محلي مركز باستخدام سلاسل كيمب وحركات النقل والمبادلة على أفضل جدول مقبول.

```text
Algorithm 6: Generational & Memetic Genetic Algorithm (ga / ga_memetic)
Input: Instance inst, FEE budget, seed, pop_size P, k, p_c, p_m, use_local_search
Output: Result(best_x, best_f, feasible, history, elapsed, evals, hparams)

1: rng <- default_rng(seed), ls_rng <- default_rng(seed + 300007), tracker <- Tracker(budget)
2: Initialize population pop_x[0..P-1] (35% DSATUR + 65% random); repair if mode == "repair"
3: Evaluate each individual (charge 1.0 FEE each) and record best_feasible_x
4: ls_reserve <- 0.06 * budget.max_evals if use_local_search else 0.0
5: while not budget.exhausted and (budget.max_evals - budget.used) > ls_reserve do
6:     Copy top elitism_count individuals to next_pop_x
7:     while next_pop_x not full and (budget.max_evals - budget.used) > ls_reserve do
8:         p1 <- tournament_select(pop_x, pop_fit, k, rng)
9:         p2 <- tournament_select(pop_x, pop_fit, k, rng)
10:        c1, c2 <- crossover_chromosomes(p1, p2, p_c, rng)
11:        Mutate c1, c2 with rate p_m; apply repair if mode == "repair"
12:        Evaluate children (charge 1.0 FEE each) and insert into next_pop_x
13:    pop_x <- next_pop_x
14:    if use_local_search and mid-budget generation reached then
15:        Refine elite solution via intensify(inst, best_feasible_x, budget, ls_rng)
16:    tracker.record(generation, best_feasible_f, diversity(pop_x))
17: if use_local_search and not budget.exhausted then
18:    best_feasible_x <- intensify(inst, best_feasible_x, budget, ls_rng)
19: return Result(best_feasible_x, best_feasible_f, feasible, tracker.to_array(), ...)
```

## 3. النتائج التجريبية الأولية ومقارنة `ga` مع `ga_memetic` (`seed = 0`)

يعرض الجدول التالي نتائج التشغيل التجريبي للنسخة الجينية النقية `ga` والنسخة الميميتية `ga_memetic` على مسائل `Tiny` ($500	ext{ FEE}$) و `Small` ($1500	ext{ FEE}$) و `Medium` ($2500	ext{ FEE}$) عند البذرة `seed = 0`:

| المسألة | $f(x^{\mathrm{planted}})$ | `ga` (النقية) | `ga_memetic` (الهجينة) | مقدار التحسن الميميتي | مقبولية الحلول ($V(x)=0$) |
|---|---:|---:|---:|---:|---:|
| `tiny-a` | 30.2333 | **23.8000** | **23.8000** | 0.0000 | 100% |
| `tiny-b` | 29.3667 | **24.8333** | **24.8333** | 0.0000 | 100% |
| `tiny-c` | 33.2500 | **29.7000** | **29.7000** | 0.0000 | 100% |
| `small-a` | 37.3833 | 32.8333 | **32.7333** | -0.1000 | 100% |
| `small-b` | 39.0433 | **32.6567** | **32.6567** | 0.0000 | 100% |
| `small-c` | 35.5667 | 32.1300 | **32.1100** | -0.0200 | 100% |
| `medium-a` | 33.3207 | 31.1073 | **30.8520** | -0.2553 | 100% |
| `medium-b` | 34.1120 | 30.9800 | **30.4653** | -0.5147 | 100% |
| `medium-c` | 35.7327 | 31.1013 | **30.7633** | -0.3380 | 100% |

تؤكد هذه النتائج ملاحظة Burke et al. (1996) حول أهمية التهجين الميميتي في جدولة الامتحانات؛ إذ تحافظ كلتا النسختين على مقبولية $100\%$ بفضل مشغّل الإصلاح، وتتفوق `ga_memetic` بشكل مطّرد كلما ازداد حجم المسألة وعدد حواف التعارض.
