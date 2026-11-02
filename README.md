# مشروع جدولة الامتحانات (Exam Timetabling Optimization)

مقارنة بين الخوارزميات الذكية (GA و PSO و ACO) وخطوط الأساس (Random Search, DSATUR, Simulated Annealing, Branch & Bound) على مسألة جدولة الامتحانات الجامعية وفق صياغة Carter.


## حالة المستودع الحالية

هذه نسخة تأسيسية من المشروع تركّز على العقد التقني المشترك واختبارات التحقق الأولية. تتضمن النسخة الحالية تمثيل مثيل المسألة، ودوال التقييم الأساسية، ومدير ميزانية FEE، وسجل التتبع، ومجموعة اختبارات للعقد. أما خوارزميات GA وPSO وACO، وخطوط الأساس، ومحلل بيانات Toronto، ومولّد المثيلات، وتشغيل التجارب وتحليل النتائج، فهي مهام موزعة على مراحل الفريق ولم تُنفّذ كلها في هذه النسخة. ملفات الإعدادات الفارغة هياكل أولية تنتظر تعبئتها من أصحاب المراحل.

في الحالة الحالية، يشغّل الأمر `./run_all.sh --quick` اختبارات العقد التقني فقط؛ ولا يعني نجاحه أن الخوارزميات أو خط التجارب الكامل أصبح جاهزاً. لا تُعرض نتائج أداء قبل تنفيذ الخوارزميات وتشغيل التجارب الموثقة.

## فكرة المشروع
لدينا $N$ امتحان و $T$ فترة زمنية و $S$ طالب مسجلين في عدة مواد. المطلوب إسناد كل امتحان إلى فترة زمنية بحيث لا يتعارض امتحانان يشترك فيهما طالب واحد في الفترة نفسها (قيد صلب)، مع مباعدة امتحانات الطالب الواحد قدر الإمكان لتقليل كلفة التقارب $f(x)$ (هدف ناعم).

تتم المقارنة بين جميع الخوارزميات باستخدام ميزانية تقييمات موحدة وعادلة (Full Evaluation Equivalent - FEE) بدلاً من عدد الأجيال أو التكرارات.

## هيكل المستودع
```text
exam-timetabling/
├── README.md
├── CONTRIBUTIONS.md
├── requirements.txt
├── pyproject.toml
├── Makefile
├── run_all.sh
├── configs/                # YAML configs for each solver (defaults + search_space)
│   └── tuned/              # Tuned configs after hyperparameter tuning
├── data/                   # instances.yaml, toronto/, generated/, checksums.txt
├── src/
│   ├── core/               # P3: instance, problem, solver (Budget, Result, Solver), tracker
│   ├── data/               # P2: toronto parser, generator, stats, validate
│   ├── baselines/          # P4: random_search, dsatur, sa, exact_bb
│   ├── local_search/       # P4: move_ls, swap_ls, kempe_ls
│   └── algorithms/         # P5/P6/P7: ga.py, pso.py, aco.py
├── experiments/            # P8: run.py, tune.py, ablations.yaml, config.yaml
├── analysis/               # P8/P9: stats.py, make_figures.py, style.py
├── results/                # raw/, processed/, figures/, tables/
├── tests/
├── docs/
└── report/
```

## إعداد البيئة والتشغيل
تتوافق نسخ المكتبات المثبّتة حالياً مع Python 3.11 وPython 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest

# Format and lint before opening a PR
make format
make lint
```

## قواعد العمل في Git للفريق
1. الفروع الأساسية هي `main` (للنسخ المستقرة عند كل Gate) و `develop` (لدمج مراحل الفريق). ممنوع الـ push المباشر على `main` و `develop`.
2. كل طالب ينشئ فرعاً خاصاً بمرحلته من `develop` بالصيغة:
   `feature/p<رقم المرحلة>-<وصف>` (مثال: `feature/p2-data-generator`).
3. قبل أول commit يجب ضبط الاسم الحقيقي والبريد ليظهر في `git shortlog -sn`:
   ```bash
   git config user.name "Your Name"
   git config user.email "your.email@example.com"
   ```
4. عنوان الـ Pull Request يكون بالصيغة: `[P5] feat(ga): add tournament selection`، والدمج في `develop` يتم بطريقة **Squash and Merge** بعد موافقة المراجع المحدد في جدول الأدوار ومرور `pytest`.
5. تفاصيل الصياغة الرياضية والتواقيع موجودة في [`docs/interface_contract.md`](docs/interface_contract.md).
