# القسم السابع من التقرير: خوارزمية مستعمرة النمل ($\mathcal{MAX}	ext{-}\mathcal{MIN}$ Ant System) والنسخة الميميتية

## 1. تصميم نظام النمل ذي الحدين الأعلى والأدنى (MMAS) لجدولة الامتحانات

تتلاءم خوارزمية مستعمرة النمل (Ant Colony Optimization - ACO) بطبيعتها البنائية مع مسائل تلوين البيانات والجدولة التوافقية (Dorigo & Stützle, 2004). اعتمدنا في `src/algorithms/aco.py` تصميم نظام النمل ذي الحدين الأعلى والأدنى ($\mathcal{MAX}	ext{-}\mathcal{MIN}$ Ant System - MMAS) الذي قدمه Stützle & Hoos (2000) وطوّره Socha, Knowles, & Samples (2002) لمسائل الجداول الجامعية.

### 1.1 تمثيل الفيرومون والمعلومة الإرشادية الآنية
يُعرَّف أثر الفيرومون $	au_{i,s}$ على الزوج (امتحان $i$، فترة $s$) ليعكس جودة إسناد الامتحان $i \in \{0, \dots, N-1\}$ إلى الفترة الزمنية $s \in \{0, \dots, T-1\}$ عبر الأجيال السابقة. وفي كل جيل، تبني كل نملة $k \in \{1, \dots, m\}$ جدولاً كاملاً خطوة بخطوة:
1. **ترتيب اختيار الامتحانات:** في كل خطوة بناء، تختار النملة الامتحان غير المجدول $u$ ذا أعلى درجة تشبع آنية (عدد الفترات المختلفة المستعملة لدى جيرانه المجدولين سلفاً)، مع كسر التعادل بالدرجة الساكنة $\deg(u)$ ثم بضجيج عشوائي طفيف لتنويع مسارات النمل داخل المستعمرة.
2. **حساب المعلومة الإرشادية $\eta_{u,s}$:** لكل فترة مرشحة $s$، تحسب النملة وزن التعارض الصلب الآني $V_{u}(s)$ وكلفة التقارب الآنية $P_{u}(s)$ مع جيران الامتحان $u$ الذين أُسندت لهم فترات في الخطوات السابقة:
$$V_{u}(s) = \sum_{v \in N(u),\; x_v = s} c_{uv}, \qquad P_{u}(s) = 
rac{1}{S} \sum_{v \in N(u),\; 1 \le |s - x_v| \le 5} c_{uv}\,2^{5 - |s - x_v|}$$
وتحصر النملة اختيارها في مجموعة الفترات ذات التعارض الأدنى $\mathcal{C}_u = rg\min_{s} V_u(s)$ (وهي الفترات الخالية من التعارض متى وُجدت)، وتعرّف المعلومة الإرشادية:
$$\eta_{u,s} = 
rac{1}{1 + P_u(s)} \quad 
orall\, s \in \mathcal{C}_u$$
3. **قاعدة الانتقال الاحتمالية:** تختار النملة الفترة $s \in \mathcal{C}_u$ بأسلوب الاستغلال الجشع باحتمال $q_0 = 0.15$، أو عبر عجلة الروليت باحتمال $1 - q_0$ وفق التوزيع:
$$p_{u,s} = 
rac{	au_{u,s}^lpha \cdot \eta_{u,s}^eta}{\sum_{r \in \mathcal{C}_u} 	au_{u,r}^lpha \cdot \eta_{u,r}^eta}$$
وطبقاً لقاعدة الميزانية العادلة في القسم 3.3، لا تُحتسب حسابات $\eta_{u,s}$ أثناء البناء ضمن ميزانية التقييمات FEE، بل يُحتسب التقييم الكامل للجدول بعد اكتمال بناء النملة بمقدار $1.0	ext{ FEE}$.

### 1.2 تحديث الفيرومون وحدود $[	au_{\min}, 	au_{\max}]$
بعد انتهاء جميع نمل الجيل من بناء جداولها، يتبخر الفيرومون بمعدل $
ho = 0.12$ ويُضاف إيداع جديد من النملة النخبوية المختارة $x^{	ext{elite}}$ (إما أفضل نملة في الجيل `ibest` أو أفضل حل شامل `gbest` كل `gbest_deposit_freq` أجيال):
$$	au_{i,s} \leftarrow 	ext{clip}\!\Big((1 - 
ho)\,	au_{i,s} + \Delta 	au_{i,s}^{	ext{elite}},\; 	au_{\min},\; 	au_{\max}\Big), \qquad \Delta 	au_{i,s}^{	ext{elite}} = egin{cases} 
rac{1}{\max(1, F(x^{	ext{elite}}))} & 	ext{if } x_i^{	ext{elite}} = s \ 0 & 	ext{otherwise} \end{cases}$$
حيث يُضبط الحد الأعلى $	au_{\max} = 
rac{1}{
ho \cdot \max(1, F(x^{	ext{gbest}}))}$ والحد الأدنى $	au_{\min}$ بدلالة معامل التقارب $p_{	ext{best}} = 0.05$، مع تطبيق تنعيم المسارات $	au \leftarrow 	au + \delta(	au_{\max} - 	au)$ عند ركود المستعمرة لعدة أجيال متتالية.

## 2. النسخة الميميتية الهجينة (`aco_memetic`)

لدمج الاستكشاف التعاوني للمستعمرة مع الاستغلال الدقيق للجوار، تزوّد فئة `ACOMemeticSolver` (`name="aco_memetic"`) نظام MMAS بمرحلتين من البحث المحلي المشترك `intensify` (`move_ls + kempe_ls + swap_ls`):
- **تكثيف منتصف الميزانية:** عند منتصف ميزانية التقييمات، يُصقل أفضل جدول عالمي `gbest` بواسطة `intensify` وتُحدَّث مصفوفة الفيرومون بالجدول المحسّن لتوجيه النمل في النصف الثاني من البحث.
- **الصقل الختامي المركز:** يُحجز هامش طفيف من نهاية الميزانية ($6\%$) لتطبيق جولات مكثفة من `move_ls` وسلاسل كيمب `kempe_ls` والمبادلات `swap_ls` على أفضل جدول مقبول توصلت إليه المستعمرة.

```text
Algorithm 5: Memetic MAX-MIN Ant System (aco / aco_memetic)
Input: Instance inst, FEE budget, seed, n_ants m, alpha, beta, rho, q0, use_local_search
Output: Result(best_x, best_f, feasible, history, elapsed, evals, hparams)

1: rng <- default_rng(seed), ls_rng <- default_rng(seed + 200003), tracker <- Tracker(budget)
2: x_seed <- dsatur_construct(inst, rng); evaluate x_seed (charge 1.0 FEE) and set gbest <- x_seed
3: Compute (tau_min, tau_max) from gbest and initialize tau[N, T] <- tau_max
4: ls_reserve <- 0.06 * budget.max_evals if use_local_search else 0.0
5: while not budget.exhausted and (budget.max_evals - budget.used) > ls_reserve do
6:     for k = 0 to m - 1 do
7:         if budget.exhausted then break
8:         x_raw <- construct_ant_timetable(inst, tau, alpha, beta, q0, rng)
9:         x_ant <- repair(inst, x_raw, rng) if constraint_handling == "repair" else x_raw
10:        Evaluate f(x_ant), V(x_ant), charge 1.0 FEE, and update ibest and gbest
11:    if use_local_search and mid-budget checkpoint reached then
12:        gbest <- intensify(inst, gbest, budget, ls_rng)
13:    Update (tau_min, tau_max) and evaporate/deposit pheromones using ibest or gbest
14:    if stagnation threshold reached then smooth tau toward tau_max
15:    tracker.record(iteration, best_feasible_f, diversity(colony_x))
16: if use_local_search and not budget.exhausted then
17:    best_feasible_x <- intensify(inst, best_feasible_x, budget, ls_rng)
18: return Result(best_feasible_x, best_feasible_f, feasible, tracker.to_array(), ...)
```

## 3. النتائج التجريبية الأولية ومقارنة `aco` مع `aco_memetic` (`seed = 0`)

يوضح الجدول التالي نتائج التشغيل التجريبي للنسخة النقية `aco` والنسخة الميميتية `aco_memetic` على مسائل `Tiny` ($500	ext{ FEE}$) و `Small` ($1500	ext{ FEE}$) و `Medium` ($2500	ext{ FEE}$) عند البذرة `seed = 0`:

| المسألة | $f(x^{\mathrm{planted}})$ | `aco` (MMAS النقية) | `aco_memetic` (الهجينة) | مقدار التحسن الميميتي | مقبولية الحلول ($V(x)=0$) |
|---|---:|---:|---:|---:|---:|
| `tiny-a` | 30.2333 | **23.8000** | **23.8000** | 0.0000 | 100% |
| `tiny-b` | 29.3667 | **24.8333** | **24.8333** | 0.0000 | 100% |
| `tiny-c` | 33.2500 | **29.7000** | **29.7000** | 0.0000 | 100% |
| `small-a` | 37.3833 | **32.6733** | **32.6733** | 0.0000 | 100% |
| `small-b` | 39.0433 | **31.9767** | **31.9767** | 0.0000 | 100% |
| `small-c` | 35.5667 | **32.0467** | **32.0467** | 0.0000 | 100% |
| `medium-a` | 33.3207 | 30.9740 | **30.2947** | -0.6793 | 100% |
| `medium-b` | 34.1120 | 31.3173 | **30.6120** | -0.7053 | 100% |
| `medium-c` | 35.7327 | 31.4293 | **31.0940** | -0.3353 | 100% |

تُبيّن النتائج أن البناء التعاوني الموجه بالفيرومون ودرجة التشبع في `aco` يصل مباشرة إلى الحلول المثلى أو شبه المثلى على المسائل الصغيرة (`Tiny` و `Small`) بمقبولية كاملة ($100\%$)، بينما يظهر الأثر الجوهري للنسخة الهجينة `aco_memetic` في المسائل المتوسطة (`Medium` ذات $120$ امتحاناً و $4500+$ حافة تعارض)، حيث يخفض البحث المحلي وسلاسل كيمب كلفة التقارب بمقدار $0.33$ إلى $0.70$ نقطة إضافية لكل طالب.
