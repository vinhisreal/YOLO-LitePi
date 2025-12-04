
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
sns.set_palette("husl")


class ResultsAnalyzer:
    """Analyze and visualize benchmark results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.raw_df = None
        self.summary_df = None
        
    def load_results(self, experiment_name: str):
        """Load raw and summary results"""
        raw_path = self.results_dir / f"{experiment_name}_raw.csv"
        summary_path = self.results_dir / f"{experiment_name}_summary.csv"
        
        self.raw_df = pd.read_csv(raw_path)
        self.summary_df = pd.read_csv(summary_path)
        
        print(f"✓ Loaded {len(self.raw_df)} records from {experiment_name}")
        
    def generate_table_4_6(self) -> str:
        """
        Generate Table for End-to-End Pipeline Performance
        (Following the paper's Table 4, 5, 6 format)
        """
        # Group by configuration
        table_data = []
        
        for config_id in self.raw_df['config_id'].unique():
            subset = self.raw_df[self.raw_df['config_id'] == config_id]
            
            # Extract model names
            detector = subset['detector_name'].iloc[0]
            det_backend = subset['detector_backend'].iloc[0]
            classifier = subset['classifier_name'].iloc[0]
            cls_backend = subset['classifier_backend'].iloc[0]
            
            # Compute statistics
            row = {
                'Detector': f"{detector.upper()}",
                'Det Backend': det_backend.upper(),
                'Classifier': classifier.capitalize(),
                'Cls Backend': cls_backend.upper(),
                'T_det (ms)': f"{subset['t_detection'].mean():.2f}±{subset['t_detection'].std():.2f}",
                'T_cls (ms)': f"{subset['t_classification'].mean():.2f}±{subset['t_classification'].std():.2f}",
                'T_roi (ms)': f"{subset['t_roi_extract'].mean():.2f}±{subset['t_roi_extract'].std():.2f}",
                'T_total (ms)': f"{subset['t_total'].mean():.2f}±{subset['t_total'].std():.2f}",
                'FPS': f"{(1000/subset['t_total']).mean():.2f}",
                'CPU (%)': f"{subset['cpu_percent'].mean():.1f}",
                'Memory (MB)': f"{subset['memory_mb'].mean():.1f}",
            }
            
            # Add accuracy if available
            if 'tp_detections' in subset.columns:
                precision = subset['tp_detections'].sum() / (subset['tp_detections'].sum() + subset['fp_detections'].sum())
                recall = subset['tp_detections'].sum() / (subset['tp_detections'].sum() + subset['fn_detections'].sum())
                row['Precision'] = f"{precision:.4f}"
                row['Recall'] = f"{recall:.4f}"
            
            table_data.append(row)
        
        df_table = pd.DataFrame(table_data)
        
        # Generate LaTeX table
        latex_table = df_table.to_latex(
            index=False,
            escape=False,
            caption="End-to-end pipeline performance on Raspberry Pi 5 (8GB). "
                   "Best results are highlighted in bold, second-best underlined.",
            label="tab:e2e_performance"
        )
        
        # Save
        output_path = self.results_dir / "table_e2e_performance.tex"
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        print(f"✓ Table saved to: {output_path}")
        print("\nPreview:")
        print(df_table.to_string(index=False))
        
        return latex_table
    
    def plot_latency_breakdown(self, save_path: str = "fig_latency_breakdown.pdf"):
        """
        Figure: Stacked bar chart showing latency breakdown
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Prepare data
        configs = self.raw_df['config_id'].unique()
        breakdown_data = {
            'Detection': [],
            'ROI Extract': [],
            'Classification': [],
            'Postprocess': []
        }
        
        labels = []
        for config in configs:
            subset = self.raw_df[self.raw_df['config_id'] == config]
            
            breakdown_data['Detection'].append(subset['t_detection'].mean())
            breakdown_data['ROI Extract'].append(subset['t_roi_extract'].mean())
            breakdown_data['Classification'].append(subset['t_classification'].mean())
            breakdown_data['Postprocess'].append(subset['t_postprocess'].mean())
            
            # Short label
            det = subset['detector_name'].iloc[0][:4]
            cls = subset['classifier_name'].iloc[0][:6]
            backend = subset['detector_backend'].iloc[0]
            labels.append(f"{det}+{cls}\n{backend}")
        
        # Create stacked bar
        x = np.arange(len(labels))
        width = 0.6
        
        bottom = np.zeros(len(labels))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for i, (stage, values) in enumerate(breakdown_data.items()):
            ax.bar(x, values, width, label=stage, bottom=bottom, color=colors[i])
            bottom += values
        
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_title('End-to-End Pipeline Latency Breakdown', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_full_path = self.results_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Latency breakdown plot saved to: {save_full_path}")
    
    def plot_accuracy_speed_tradeoff(self, save_path: str = "fig_accuracy_speed_tradeoff.pdf"):
        """
        Figure: Scatter plot showing accuracy vs speed tradeoff
        """
        if 'tp_detections' not in self.raw_df.columns:
            print("⚠ Accuracy data not available, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Compute metrics per config
        plot_data = []
        for config in self.raw_df['config_id'].unique():
            subset = self.raw_df[self.raw_df['config_id'] == config]
            
            # Accuracy (F1-score)
            tp = subset['tp_detections'].sum()
            fp = subset['fp_detections'].sum()
            fn = subset['fn_detections'].sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Speed
            fps = (1000 / subset['t_total']).mean()
            
            # Label
            det = subset['detector_name'].iloc[0]
            cls = subset['classifier_name'].iloc[0]
            backend = subset['detector_backend'].iloc[0]
            
            plot_data.append({
                'f1': f1,
                'fps': fps,
                'label': f"{det}+{cls}",
                'backend': backend
            })
        
        # Plot
        df_plot = pd.DataFrame(plot_data)
        
        # Color by backend
        backend_colors = {'onnx': 'red', 'openvino': 'blue', 'ncnn': 'green'}
        
        for backend in df_plot['backend'].unique():
            subset = df_plot[df_plot['backend'] == backend]
            ax.scatter(subset['fps'], subset['f1'], 
                      s=150, alpha=0.7, 
                      color=backend_colors.get(backend, 'gray'),
                      label=backend.upper(), 
                      edgecolors='black', linewidth=1)
            
            # Annotate points
            for _, row in subset.iterrows():
                ax.annotate(row['label'], 
                           (row['fps'], row['f1']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Throughput (FPS)', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
        ax.legend(title='Backend', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_full_path = self.results_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Accuracy-speed tradeoff plot saved to: {save_full_path}")
    
    def plot_resource_utilization(self, save_path: str = "fig_resource_utilization.pdf"):
        """
        Figure: CPU and Memory utilization comparison
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        configs = self.raw_df['config_id'].unique()
        labels = []
        cpu_means = []
        cpu_stds = []
        mem_means = []
        mem_stds = []
        
        for config in configs:
            subset = self.raw_df[self.raw_df['config_id'] == config]
            
            # Short label
            det = subset['detector_name'].iloc[0][:5]
            cls = subset['classifier_name'].iloc[0][:6]
            labels.append(f"{det}+{cls}")
            
            cpu_means.append(subset['cpu_percent'].mean())
            cpu_stds.append(subset['cpu_percent'].std())
            mem_means.append(subset['memory_mb'].mean())
            mem_stds.append(subset['memory_mb'].std())
        
        x = np.arange(len(labels))
        width = 0.6
        
        # CPU plot
        ax1.bar(x, cpu_means, width, yerr=cpu_stds, 
               capsize=5, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.set_ylabel('CPU Utilization (%)', fontsize=11)
        ax1.set_xlabel('Configuration', fontsize=11)
        ax1.set_title('CPU Usage', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # Memory plot
        ax2.bar(x, mem_means, width, yerr=mem_stds,
               capsize=5, alpha=0.7, color='#e74c3c', edgecolor='black')
        ax2.set_ylabel('Memory Usage (MB)', fontsize=11)
        ax2.set_xlabel('Configuration', fontsize=11)
        ax2.set_title('Memory Consumption', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_full_path = self.results_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Resource utilization plot saved to: {save_full_path}")
    
    def plot_latency_distribution(self, save_path: str = "fig_latency_distribution.pdf"):
        """
        Figure: Box plot showing latency distribution across configurations
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        configs = self.raw_df['config_id'].unique()
        data_to_plot = []
        labels = []
        
        for config in configs:
            subset = self.raw_df[self.raw_df['config_id'] == config]
            data_to_plot.append(subset['t_total'].values)
            
            # Short label
            det = subset['detector_name'].iloc[0][:5]
            backend = subset['detector_backend'].iloc[0]
            labels.append(f"{det}-{backend}")
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       meanprops=dict(color='green', linewidth=2, linestyle='--'))
        
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_title('End-to-End Latency Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        
        # Add real-time threshold line
        ax.axhline(y=100, color='r', linestyle=':', linewidth=2, 
                  label='Real-time threshold (100ms)', alpha=0.7)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        save_full_path = self.results_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Latency distribution plot saved to: {save_full_path}")
    
    def plot_fps_comparison(self, save_path: str = "fig_fps_comparison.pdf"):
        """
        Figure: Bar chart comparing FPS across configurations
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        configs = self.raw_df['config_id'].unique()
        labels = []
        fps_means = []
        fps_stds = []
        colors = []
        
        for config in configs:
            subset = self.raw_df[self.raw_df['config_id'] == config]
            
            fps = 1000 / subset['t_total']
            fps_means.append(fps.mean())
            fps_stds.append(fps.std())
            
            # Label
            det = subset['detector_name'].iloc[0]
            cls = subset['classifier_name'].iloc[0][:8]
            backend = subset['detector_backend'].iloc[0]
            labels.append(f"{det}+{cls}\n({backend})")
            
            # Color by backend
            if backend == 'ncnn':
                colors.append('#2ecc71')
            elif backend == 'openvino':
                colors.append('#3498db')
            else:
                colors.append('#e74c3c')
        
        x = np.arange(len(labels))
        bars = ax.bar(x, fps_means, yerr=fps_stds, capsize=5, 
                     alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, fps_means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Throughput (FPS)', fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_title('End-to-End Pipeline Throughput Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        
        # Add real-time threshold
        ax.axhline(y=7, color='orange', linestyle='--', linewidth=2, 
                  label='Target: 7 FPS', alpha=0.7)
        ax.axhline(y=10, color='green', linestyle='--', linewidth=2,
                  label='Ideal: 10 FPS', alpha=0.7)
        
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_full_path = self.results_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ FPS comparison plot saved to: {save_full_path}")
    
    def generate_all_figures(self):
        """Generate all figures for the paper"""
        print("\n" + "="*60)
        print("Generating all figures...")
        print("="*60 + "\n")
        
        self.plot_latency_breakdown()
        self.plot_fps_comparison()
        self.plot_latency_distribution()
        self.plot_resource_utilization()
        self.plot_accuracy_speed_tradeoff()
        
        print("\n✓ All figures generated successfully!")
    
    def generate_summary_report(self, save_path: str = "summary_report.txt"):
        """Generate text summary report"""
        report = []
        report.append("="*60)
        report.append("END-TO-END PIPELINE BENCHMARK SUMMARY")
        report.append("="*60)
        report.append("")
        
        # Overall statistics
        report.append("Overall Statistics:")
        report.append(f"  Total iterations: {len(self.raw_df)}")
        report.append(f"  Number of configurations: {self.raw_df['config_id'].nunique()}")
        report.append("")
        
        # Best configuration
        fps_by_config = {}
        for config in self.raw_df['config_id'].unique():
            subset = self.raw_df[self.raw_df['config_id'] == config]
            avg_fps = (1000 / subset['t_total']).mean()
            fps_by_config[config] = avg_fps
        
        best_config = max(fps_by_config, key=fps_by_config.get)
        best_fps = fps_by_config[best_config]
        
        report.append("Best Configuration:")
        report.append(f"  {best_config}")
        report.append(f"  Throughput: {best_fps:.2f} FPS")
        report.append("")
        
        # Configuration comparison
        report.append("Configuration Comparison (sorted by FPS):")
        report.append("")
        sorted_configs = sorted(fps_by_config.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (config, fps) in enumerate(sorted_configs, 1):
            subset = self.raw_df[self.raw_df['config_id'] == config]
            latency = subset['t_total'].mean()
            cpu = subset['cpu_percent'].mean()
            mem = subset['memory_mb'].mean()
            
            report.append(f"{rank}. {config}")
            report.append(f"   FPS: {fps:.2f}, Latency: {latency:.2f}ms, CPU: {cpu:.1f}%, Memory: {mem:.1f}MB")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save
        output_path = self.results_dir / save_path
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Summary report saved to: {output_path}")



def main():
    """Example usage"""
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(results_dir="results_e2e_pipeline")
    
    # Load results
    analyzer.load_results(experiment_name="end_to_end_comparison")
    
    # Generate table for paper
    print("\n" + "="*60)
    print("Generating LaTeX table...")
    print("="*60)
    analyzer.generate_table_4_6()
    
    # Generate all figures
    analyzer.generate_all_figures()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()