import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon

def analyze_results(csv_path: str):
    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. ملخص سريع لكل خوارزمية
    print("\n=== Summary by Algorithm ===")
    summary = df.groupby('algorithm').agg(
        mean_f=('best_f', 'mean'),
        median_f=('best_f', 'median'),
        std_f=('best_f', 'std'),
        feasible_rate=('feasible', 'mean'),
        avg_time=('elapsed', 'mean'),
        avg_evals=('evals_used', 'mean')
    ).reset_index()
    
    print(summary.to_string(index=False))
    
    # 2. إنشاء جدول تفصيلي لكل (خوارزمية، نسخة)
    print("\n=== Detailed Results by Instance ===")
    detailed = df.groupby(['algorithm', 'instance']).agg(
        mean_f=('best_f', 'mean'),
        median_f=('best_f', 'median'),
        feasible_rate=('feasible', 'mean')
    ).reset_index()
    print(detailed.to_string(index=False))
    
    # 3. التحليل الإحصائي (Friedman Test)
    # نقوم بمقارنة الخوارزميات على كل نسخة (نتجاهل القيم المفقودة)
    # ملاحظة: هذا يتطلب أن تكون كل الخوارزميات قد جربت على نفس البذور
    print("\n=== Statistical Tests ===")
    
    # نأخذ نسخة واحدة كمثال (tiny-a)
    example_inst = df['instance'].iloc[0]
    inst_df = df[df['instance'] == example_inst]
    
    # تحويل البيانات لتصبح كل خوارزمية في عمود
    pivot_df = inst_df.pivot(index='seed', columns='algorithm', values='best_f')
    
    # إزالة الصفوف التي تحتوي على قيم مفقودة (NaN)
    pivot_df = pivot_df.dropna()
    
    if pivot_df.shape[1] >= 3:
        try:
            stat, p_val = friedmanchisquare(*[pivot_df[col].values for col in pivot_df.columns])
            print(f"Friedman Test on {example_inst}:")
            print(f"  Statistic: {stat:.4f}")
            print(f"  p-value: {p_val:.4e}")
            if p_val < 0.05:
                print("  Result: Significant differences between algorithms (p < 0.05).")
            else:
                print("  Result: No significant differences detected.")
        except Exception as e:
            print(f"  Could not run Friedman test: {e}")
    else:
        print("  Not enough algorithms to run Friedman test.")

    # 4. حفظ الجداول في مجلد results/tables
    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary.to_csv(out_dir / 'summary_by_algorithm.csv', index=False)
    detailed.to_csv(out_dir / 'detailed_by_instance.csv', index=False)
    
    print(f"\nTables saved to {out_dir}/")

if __name__ == "__main__":
    # تأكد من أن اسم الملف يطابق ما أنتجه run.py
    analyze_results('results/raw/full_run.csv')