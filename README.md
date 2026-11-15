# مشروع جدولة الامتحانات الجامعية (Exam Timetabling Optimization)

تطبيق ومقارنة تجريبية بين الخوارزميات الذكية (**GA** و **PSO** و **ACO**) ونسخها الهجينة (**Memetic Algorithms**)، بالإضافة إلى خطوط الأساس (**Random Search**, **DSATUR**, **Simulated Annealing**, **Exact Branch & Bound**) لحل مسألة جدولة الامتحانات الجامعية وفق صياغة **Carter**[cite: 38].

---

## طريقة تشغيل الخوارزميات

يمكن تشغيل المشروع وتجربة جميع الخوارزميات بإحدى طريقتين:

### الطريقة الأولى: التشغيل المباشر عبر Google Colab (موصى بها)
لا تتطلب هذه الطريقة أي تثبيت على الجهاز؛ حيث يقوم دفتر العمل باستنساخ المستودع، وتثبيت المكتبات، وتشغيل الاختبارات، وتنفيذ الخوارزميات على جميع مجموعات البيانات تلقائياً:

- **رابط التشغيل المباشر على Google Colab:**  
  https://colab.research.google.com/drive/1p7etv_X4_jQOWhcI6i8vBHjCyBSvDu5K?usp=sharing
- **ملف الدفتر المرفق:** `Exam_Timetabling_Execution.ipynb`

**خطوات التشغيل:**
1. افتح رابط Google Colab أعلاه (أو ارفع ملف `Exam_Timetabling_Execution.ipynb` المرفق إلى موقع [Google Colab](https://colab.research.google.com/)).
2. من القائمة العلوية اختر **Runtime -> Run all** (أو شغّل الخلايا البرمجية بالترتيب من الأعلى إلى الأسفل).
3. ستظهر في الخلايا الأخيرة جداول النتائج المقارنة (قيمة دالة الهدف `best_f`، نسبة الجدوى `Feasibility`، رصيد `FEE` المستهلك، وزمن التنفيذ)، يليها التحليل الإحصائي باختباري **Friedman** و **Wilcoxon-Holm**.

---

### الطريقة الثانية: التشغيل المحلي عبر سطر الأوامر (CLI)
يتطلب التشغيل المحلي وجود **Python 3.10+**[cite: 38]:

1. **استنساخ المستودع وتثبيت المكتبات:**
   ```bash
   git clone [https://github.com/Walid2003H/Exam-timetabling.git](https://github.com/Walid2003H/Exam-timetabling.git)
   cd Exam-timetabling
   pip install -r requirements.txt
   ```

2. **التحقق من سلامة الكود واختبارات الوحدة (42 اختباراً):**
   ```bash
   pytest tests/ -q
   ```

3. **تشغيل التجارب على مجموعات البيانات (`Tiny`, `Small`, `Medium`, `Large`):**
   ```bash
   # تشغيل التحقق السريع لجميع الخوارزميات على المسائل الـ 12
   python -m experiments.run --mode quick

   # تشغيل الوضع التجريبي المتوسط (5 بذور)
   python -m experiments.run --mode pilot

   # تشغيل التجربة الموسعة (30 بذرة)
   python -m experiments.run --mode full

   # أو تشغيل خط العمل الكامل (اختبارات + تجارب + تحليل إحصائي)
   bash run_all.sh
   ```

4. **تشغيل خوارزمية محددة برمجياً (مثال بلغة Python):**
   ```python
   from src.core.instance import Instance
   from src.core.budget import Budget
   from src.algorithms.ga import solve_ga_memetic  # أو pso_memetic / aco_memetic

   inst = Instance.load("data/generated/small-a.npz")
   budget = Budget(max_fee=1500)
   result = solve_ga_memetic(inst, budget=budget, seed=42)

   print(f"Feasible: {result.feasible} | Cost f(x): {result.best_f:.4f} | FEE Used: {result.evals}")
   ```

---

## فكرة المشروع ومنهجية التقييم
لدينا $N$ امتحان و $T$ فترة زمنية و $S$ طالب مسجلين في عدة مواد. المطلوب إسناد كل امتحان إلى فترة زمنية بحيث لا يتعارض امتحانان يشترك فيهما طالب واحد في الفترة نفسها (**قيد صلب $V(x) = 0$**)، مع مباعدة امتحانات الطالب الواحد قدر الإمكان لتقليل كلفة التقارب $f(x)$ وفق أوزان Carter (**هدف ناعم**)[cite: 38].

لضمان عدالة المقارنة بين الخوارزميات المختلفة، تعتمد المنظومة ميزانية تقييمات موحدة تُقاس بوحدة مكافئ التقييم الكامل (**Full Evaluation Equivalent - FEE**) بدلاً من تثبيت عدد الأجيال أو التكرارات، مع تسريع حسابات الرسم البياني عبر تمثيل **CSR** ومترجم **Numba JIT**[cite: 38].

---

## هيكل المستودع
```text
Exam-timetabling/
├── README.md               # دليل المشروع وطريقة التشغيل
├── requirements.txt        # المكتبات المطلوبة للتشغيل
├── pyproject.toml          # إعدادات المشروع وأدوات الفحص (pytest / ruff)
├── run_all.sh              # سكربت التشغيل الشامل
├── configs/                # ملفات إعدادات المعاملات لكل خوارزمية (YAML)
├── data/                   # مسائل الاختبار التركيبية وبصمات التحقق SHA-256
├── src/
│   ├── core/               # تمثيل CSR، دالة Carter، القيود، ومحاسبة ميزانية FEE
│   ├── data/               # مولّد المسائل، قارئ بيانات Toronto، والتحقق من السلامة
│   ├── baselines/          # Random Search, DSATUR, Simulated Annealing, Exact B&B
│   ├── local_search/       # مشغلات البحث المحلي (Move, Swap, Kempe-chain, Hybrid)
│   └── algorithms/         # الخوارزميات الأساسية والهجينة (GA, PSO, ACO)
├── experiments/            # منفّذ التجارب الموحد (run.py) وضبط المعاملات (tune.py)
├── analysis/               # الاختبارات الإحصائية (Friedman & Wilcoxon-Holm) والرسوم
└── tests/                  # اختبارات الوحدة والتكامل (42 اختباراً)
```
