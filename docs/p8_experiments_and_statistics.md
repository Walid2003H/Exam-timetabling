# توثيق المرحلة الثامنة (P8: Experiments, Tuning, Ablations & Statistical Analysis)

## 1. ما الذي تم بناؤه في هذه المرحلة؟
تجمع هذه المرحلة جميع الحلّالات العشرة (`random`, `dsatur`, `sa`, `exact`, `ga`, `ga_memetic`, `pso`, `pso_memetic`, `aco`, `aco_memetic`) ضمن خط تجارب وضبط وإحصاء موحد وقابل لإعادة الإنتاج بالكامل:
- `experiments/tune.py` و `configs/tuned/*.yaml`: ضبط المعاملات الفائقة (Hyperparameter Tuning) لكل خوارزمية حصرياً على مسائل الضبط المستقلة (`tune-small`, `tune-medium`, `tune-large`) ببذور الضبط `1000..1004` لمنع تسرب البيانات (Data Leakage) إلى مجموعة الاختبار، وحفظ الإعدادات الفائزة في `configs/tuned/<name>.yaml` وجدول `results/tables/tuning_summary.csv`.
- `experiments/config.yaml` و `experiments/ablations.yaml`: ملفات توصيف التجارب الرئيسية (12 مسألة عبر 4 أحجام $	imes$ 10 خوارزميات $	imes$ البذور العشوائية) وتوصيف دراسات الاستئصال الثلاث (`constraint_handling`: الإصلاح مقابل العقوبة، `memetic_local_search`: النسخ النقية مقابل الهجينة، `pso_encoding`: المفاتيح العشوائية مقابل السرب المتقطع المباشر).
- `experiments/run.py`: مشغّل التجارب الموحد الذي يتحقق مستقلاً من مقبولية الحلول ($V(x)=0$)، ويقيس زمن التنفيذ الفعلي (`elapsed`) وذروة استهلاك الذاكرة (`peak_memory_mb`)، ويحفظ السجلات الخام في `results/raw/benchmark_runs.csv` و `results/raw/ablation_runs.csv`، كما يستيفئ منحنيات التقارب والتنوع على شبكة موحدة في `results/processed/convergence_trajectories.npz`.
- `analysis/stats.py`: وحدة التحليل الإحصائي التي تحسب المتوسط والانحراف المعياري والوسيط والمدى الربيعي `IQR` وأفضل قيمة للحلول المقبولة فقط (`feasible == True`)، ومعدل المقبولية `feasible_rate`، وفجوة الأمثلية `optimality_gap_pct` على مجموعة `Tiny`، وتجري اختبار **Friedman** غير المعلمي متبوعاً باختبارات **Wilcoxon Signed-Rank** الزوجية مع تصحيح **Holm-Bonferroni** لتعدد المقارنات.
- `analysis/style.py` و `analysis/make_figures.py`: التوليد الآلي لمخططات الصندوق (`boxplot_<instance>.png`) ومخطط المقايضة بين الجودة والسرعة (`quality_vs_speed.png`) في `results/figures/`.
- `run_all.sh` و `tests/test_experiments.py`: سكربت التشغيل الشامل (`./run_all.sh --quick`) واختبارات الوحدة للمرحلة الثامنة.

## 2. كيفية التشغيل والفحص
```bash
# 1. Run all 8 unit test suites (P1 through P8)
pytest -v

# 2. Run the complete quick end-to-end pipeline (Gate 4 check)
./run_all.sh --quick

# 3. Run individual P8 stages manually
PYTHONPATH=. python3 -m experiments.tune --quick
PYTHONPATH=. python3 -m experiments.run --quick
PYTHONPATH=. python3 -m analysis.stats
PYTHONPATH=. python3 -m analysis.make_figures
```

## 3. القرارات التصميمية المهمة وأسبابها
1. **فصل مسائل وبذور الضبط عن مسائل وبذور التقييم النهائي:**
   استُخدمت المسائل `tune-*` والبذور `1000..1004` في `experiments/tune.py` بينما استُخدمت المسائل الـ 12 (`tiny-a..c`, `small-a..c`, `medium-a..c`, `large-a..c`) والبذور `0..29` في `experiments/run.py`، التزاماً بالقسم 3.4 من العقد التقني ومنعاً للانحياز التفاؤلي.
2. **تطبيق تصحيح Holm-Bonferroni بعد اختبار Friedman:**
   بما أن مقارنة عدة خوارزميات عبر اختبارات Wilcoxon المتعددة ترفع خطأ النوع الأول (Type I Error)، اعتمدنا إجراء Friedman الشامل أولاً عبر المسائل الـ 12 ثم تصحيح القيم الاحتمالية $p$ بطريقة Holm التنازلية.
