# القسم السادس من التقرير: خوارزمية تحسين سرب الجسيمات (PSO) والنسخة الميميتية الهجينة

## 1. تكييف تحسين سرب الجسيمات مع الجدولة التوافقية

صُممت خوارزمية تحسين سرب الجسيمات (Particle Swarm Optimization - PSO) التي قدمها Kennedy & Eberhart (1995) أصلاً لفضاءات البحث المستمرة $\mathbb{R}^N$. ولإسقاطها على مسألة جدولة الامتحانات التوافقية ذات القيود الصلبة، نفّذنا في `src/algorithms/pso.py` مقاربتين قياسيتين للمقارنة التجريبية:

### 1.1 الترميز الأول: المفاتيح العشوائية المستمرة (`encoding="random_keys"`)
استناداً إلى مقاربة Bean (1994)، يحتفظ كل جسيم $k \in \{1, \dots, K\}$ بمتجه موقع مستمر $z^{(k)} \in [0, 1]^N$ ومتجه سرعة $v^{(k)} \in [-v_{\max}, v_{\max}]^N$. في كل تكرار $t$، تُحدَّث السرعة والموقع وفق المعادلتين القياسيتين:
$$v_i^{(k)}(t+1) = 	ext{clip}\!\Big(w(t)\,v_i^{(k)}(t) + c_1 r_{1,i}ig(p_i^{(k)} - z_i^{(k)}(t)ig) + c_2 r_{2,i}ig(g_i - z_i^{(k)}(t)ig),\; -v_{\max},\; v_{\max}\Big)$$
$$z_i^{(k)}(t+1) = 	ext{clip}\!\Big(z_i^{(k)}(t) + v_i^{(k)}(t+1),\; 0,\; 1\Big)$$
حيث $p^{(k)}$ أفضل موقع شخصي للجسيم $k$، و $g$ أفضل موقع عالمي في السرب، و $r_1, r_2 \sim U(0,1)^N$، ويتناقص وزن العطالة $w(t)$ خطياً من $w_{\max}=0.78$ إلى $w_{\min}=0.42$ مع استهلاك ميزانية التقييمات.

ولتحويل المتجه المستمر $z^{(k)}$ إلى جدول امتحانات متقطع $x^{(k)} \in \{0, \dots, T-1\}^N$، تحسب دالة `decode_random_keys` أولوية جدولة مركّبة لكل امتحان تجمع بين درجته الساكنة المطبّعة في بيان التعارض ومفتاحه المستمر:
$$\pi_i^{(k)} = 0.65\,
rac{\deg(i)}{\max_j \deg(j)} + 0.35\,z_i^{(k)}$$
ثم تُرتَّب الامتحانات تنازلياً حسب $\pi_i^{(k)}$ ويُسند كل امتحان إلى الفترة التي تحقق أقل تعارض صلب ثم أقل كلفة تقارب مع جيرانه المجدولين سلفاً، يليه تطبيق مشغّل الإصلاح `repair` عند الحاجة. ووفق قاعدة الميزانية العادلة في القسم 3.3، لا تُحتسب عملية فك الترميز الجشعة ضمن FEE بل يُحتسب التقييم الكامل للجدول الناتج بـ $1.0	ext{ FEE}$ وينعكس زمن فك الترميز في الـ Wall-Clock Time.

### 1.2 الترميز الثاني: السرب المتقطع المباشر (`encoding="direct_discrete"`)
في هذا الترميز، يتحرك الجسيم مباشرة في فضاء الجداول $\{0, \dots, T-1\}^N$، ويحتفظ بمتجه احتمالات تحرك $q^{(k)} \in [0.05, 0.95]^N$ يُحدَّث بدلالة مؤشرات الاختلاف عن أفضل موقع شخصي وعالمي:
$$q_i^{(k)}(t+1) = 	ext{clip}\!\left(
rac{w\,q_i^{(k)}(t) + c_1 r_{1,i}\,\mathbb{I}[x_i^{	ext{pbest},(k)} 
eq x_i^{(k)}] + c_2 r_{2,i}\,\mathbb{I}[x_i^{	ext{gbest}} 
eq x_i^{(k)}]}{w + c_1 + c_2},\; 0.05,\; 0.95
ight)$$
وعند تحقق التحرك للامتحان $i$ باحتمال $q_i^{(k)}(t+1)$، يرث الجسيم فترة الامتحان $i$ من $x^{	ext{gbest}}$ باحتمال $
rac{c_2}{c_1 + c_2}$ أو من $x^{	ext{pbest},(k)}$ باحتمال $
rac{c_1}{c_1 + c_2}$، مع طفرة عشوائية طفيفة بمعدل $p_{	ext{mut}} = 0.03$ لمنع انهيار تنوع السرب.

## 2. النسخة الميميتية الهجينة (`pso_memetic`) والكتابة اللاماركية

تعاني خوارزميات الأسراب النقية في المسائل التوافقية من ضعف قدرتها على الضبط الدقيق للجوار المحلي (Fine-grained exploitation). لمعالجة ذلك، تدمج فئة `PSOMemeticSolver` (`name="pso_memetic"`) حركة السرب مع وحدة البحث المحلي المشتركة `intensify` (`move_ls + kempe_ls + swap_ls`):
- في نهاية كل دورة للسرب، يُختار أفضل جسيم في السرب (`ls_top_k = 1`) ويُحسَّن محلياً عبر `intensify` مع خصم جميع الحركات التفاضلية من نفس ميزانية `Budget`.
- عند عثور البحث المحلي على جدول أفضل $x^{	ext{ref}}$، يُعتمد التعلم اللاماركي (Lamarckian Inheritance) بتحديث `pbest_x` وإعادة ترميز الجدول المحسّن إلى الفضاء المستمر عبر `encode_slots_to_keys` لجذب بقية جسيمات السرب نحو الحوض المحسّن.

```text
Algorithm 4: Memetic Particle Swarm Optimization (pso / pso_memetic)
Input: Instance inst, FEE budget, seed, swarm_size K, encoding, use_local_search
Output: Result(best_x, best_f, feasible, history, elapsed, evals, hparams)

1: rng <- default_rng(seed), tracker <- Tracker(budget)
2: Initialize K particles (35% via DSATUR + 65% via decoded random keys / discrete uniform)
3: Repair and evaluate each particle (charging 1.0 FEE per particle); initialize pbest and gbest
4: while not budget.exhausted do
5:     w <- w_max - (w_max - w_min) * (budget.used / budget.max_evals)
6:     for k = 0 to K - 1 do
7:         if budget.exhausted then break
8:         if encoding == "random_keys" then
9:             Update velocity v[k] and continuous keys z[k]; x_cand <- decode_random_keys(inst, z[k])
10:        else
11:            x_cand, vel_prob[k] <- update_direct_discrete_particle(pop_x[k], pbest_x[k], gbest_x)
12:        x_eval <- repair(inst, x_cand, rng) if constraint_handling == "repair" else x_cand
13:        Evaluate f(x_eval), V(x_eval), charge 1.0 FEE, and update pbest[k] and gbest
14:    if use_local_search and not budget.exhausted then
15:        Refine top-k personal bests via intensify(inst, pbest_x[idx], budget, rng)
16:        Apply Lamarckian write-back to pbest[idx] and gbest
17:    tracker.record(iteration, best_feasible_f, diversity(pop_x))
18: return Result(best_feasible_x, best_feasible_f, feasible, tracker.to_array(), ...)
```

## 3. النتائج التجريبية الأولية ومقارنة الترميزين والنسخة الميميتية (`seed = 0`)

يبيّن الجدول التالي مقارنة الأداء بين الترميزين (`random_keys` و `direct_discrete`) للنسخة النقية `pso` مقابل النسخة الهجينة `pso_memetic` على مسائل `Tiny` ($500	ext{ FEE}$) و `Small` ($1500	ext{ FEE}$) و `Medium` ($2500	ext{ FEE}$) عند البذرة `seed = 0`:

| المسألة | $f(x^{\mathrm{planted}})$ | `pso` (`random_keys`) | `pso` (`direct_discrete`) | `pso_memetic` | مقبولية الحلول ($V(x)=0$) |
|---|---:|---:|---:|---:|---:|
| `tiny-a` | 30.2333 | 24.6500 | 23.8000 | **23.6333** | 100% |
| `tiny-b` | 29.3667 | **24.8333** | 24.9333 | **24.8333** | 100% |
| `tiny-c` | 33.2500 | **29.7000** | **29.7000** | 29.8000 | 100% |
| `small-a` | 37.3833 | 33.2467 | 32.9767 | **32.4733** | 100% |
| `small-b` | 39.0433 | 32.7267 | 32.1267 | **31.9767** | 100% |
| `small-c` | 35.5667 | **31.8267** | 32.1900 | 31.9600 | 100% |
| `medium-a` | 33.3207 | 30.7053 | 31.2520 | **29.4673** | 100% |
| `medium-b` | 34.1120 | 30.9580 | 30.1460 | **29.8800** | 100% |
| `medium-c` | 35.7327 | 31.2980 | 30.8460 | **30.2167** | 100% |

تُظهر النتائج أن كلاً من ترميز المفاتيح العشوائية (`random_keys`) والترميز المتقطع المباشر (`direct_discrete`) يحققان مقبولية كاملة ($100\%$) ويتفوقان بوضوح على خطوط الأساس البنائية (`dsatur` و `random`)، بينما تحقق النسخة الميميتية `pso_memetic` أفضل كلفة تقارب على المسائل المتوسطة (`Medium`) بفضل قدرة سلاسل كيمب والبحث المحلي على تفكيك الاختناقات المحلية التي يعجز عنها تحرك الجسيمات المنفرد.
