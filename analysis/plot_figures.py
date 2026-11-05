import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_plots():
    raw_dir = Path('results/raw')
    # البحث عن أحدث ملف نتائج
    csv_files = sorted(raw_dir.glob('run_*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not csv_files:
        print("No CSV files found in results/raw/")
        return
        
    latest_file = csv_files[0]
    print(f"Generating plots from {latest_file.name}...")
    df = pd.read_csv(latest_file)
    
    out_dir = Path('results/figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. رسم Boxplot لكل نسخة (Instance)
    instances = df['instance'].unique()
    for inst in instances:
        inst_df = df[df['instance'] == inst]
        
        data = []
        labels = []
        for algo in inst_df['algorithm'].unique():
            algo_data = inst_df[inst_df['algorithm'] == algo]['best_f'].values
            data.append(algo_data)
            labels.append(algo)
            
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, tick_labels=labels, patch_artist=True)
        plt.title(f'Best Fitness Distribution on {inst} (10k FEE)')
        plt.ylabel('Best Fitness (f)')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(out_dir / f'boxplot_{inst}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved boxplot for {inst}")
        
    # 2. رسم Bar Chart للمتوسط (Mean) و الوقت (Time)
    plt.figure(figsize=(10, 5))
    summary = df.groupby('algorithm').agg(mean_f=('best_f', 'mean'), avg_time=('elapsed', 'mean')).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # رسم المتوسط
    ax1.bar(summary['algorithm'], summary['mean_f'], color='skyblue', alpha=0.7, label='Mean Best Fitness')
    ax1.set_ylabel('Mean Best Fitness (f)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # رسم الوقت
    ax2.plot(summary['algorithm'], summary['avg_time'], color='red', marker='o', label='Average Time (s)')
    ax2.set_ylabel('Average Time (seconds)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Quality vs. Speed Trade-off')
    fig.tight_layout()
    plt.savefig(out_dir / 'quality_vs_speed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved quality_vs_speed chart")
    
    print(f"\nAll figures saved to {out_dir}/")

if __name__ == "__main__":
    generate_plots()