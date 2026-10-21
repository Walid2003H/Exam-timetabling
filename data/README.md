# دليل البيانات

يحتوي هذا المجلد على مواصفات المسائل وبيانات المرحلة الثانية P2.

- `instances.yaml`: أبعاد المجموعات والبذور الثابتة وميزانيات FEE المقترحة. يحدد قسم `toronto` أسماء المثيلات وعدد الفترات، لكنه لا يضم ملفات Toronto الأصلية.
- `generated/`: 15 ملفاً مضغوطاً بصيغة `.npz`؛ تشمل ثلاث مسائل لكل حجم من `tiny` و`small` و`medium` و`large` وثلاث مسائل مستقلة للضبط (`tuning`).
- `checksums.txt`: بصمات SHA-256 لجميع ملفات `.npz` الموجودة في `generated/`.
- `toronto/`: ملفات المثال `sample_toy.crs` و`sample_toy.stu` لاختبار المحلل. هذه بيانات اختبار صغيرة وليست جزءاً من معيار Toronto الأصلي.

## تحميل مسألة والتحقق منها

```python
from src.data import load_instance, validate_instance

instance = load_instance("medium-a")
validate_instance(instance, check_planted=True)
print(instance.n_exams, instance.n_slots, instance.n_edges)
```

يحمّل `load_instance` الملفات من `data/generated/` افتراضياً، لذلك نقوم بتشغيل المثال من جذر المستودع.

## إعادة التوليد والتحقق من البصمات

```bash
python - <<'PY'
from src.data import generate_all_from_yaml, write_checksums, verify_checksums

generate_all_from_yaml("data/instances.yaml", "data/generated")
write_checksums("data/generated", "data/checksums.txt")
assert verify_checksums("data/generated", "data/checksums.txt")
PY
```
