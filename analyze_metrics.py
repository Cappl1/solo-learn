#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze difficulty metrics produced during Temporal I-JEPA training.
"""

import os
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

class MetricsAnalyzer:
    def __init__(self, metrics_dir: str, output_dir: Optional[str] = None):
        """Initialize analyzer with directory containing metrics files.
        
        Args:
            metrics_dir: Directory containing the metrics_epoch_X.parquet files
            output_dir: Where to save analysis results (defaults to metrics_dir/analysis)
        """
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path(output_dir) if output_dir else self.metrics_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all metrics files
        self.metric_files = sorted(
            self.metrics_dir.glob("metrics_epoch_*.parquet"),
            key=lambda x: int(x.stem.split("_")[-1])
        )
        
        if not self.metric_files:
            raise FileNotFoundError(f"No metrics files found in {metrics_dir}")
            
        print(f"Found {len(self.metric_files)} metric files")
        
        # Load metrics into dataframes
        self.epoch_metrics = {}
        for file in self.metric_files:
            epoch = int(file.stem.split("_")[-1])
            self.epoch_metrics[epoch] = pd.read_parquet(file)
            
        # Get list of available metrics
        self.metric_cols = [
            col for col in self.epoch_metrics[0].columns 
            if col not in ["sample_id"] and not col.endswith("_quartile")
        ]
        
        print(f"Available metrics: {', '.join(self.metric_cols)}")
        
    def analyze_single_epoch(self, epoch: int = 0):
        """Analyze metrics for a single epoch.
        
        Args:
            epoch: Epoch number to analyze
        """
        if epoch not in self.epoch_metrics:
            raise ValueError(f"No metrics for epoch {epoch}")
            
        df = self.epoch_metrics[epoch]
        
        # Create correlation matrix
        corr = df[self.metric_cols].corr(method="spearman")
        
        # Save correlation matrix
        corr.to_csv(self.output_dir / f"epoch_{epoch}_correlations.csv")
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Metric Correlations (Epoch {epoch})")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch}_correlations.png", dpi=300)
        plt.close()
        
        # Plot distributions
        fig, axes = plt.subplots(len(self.metric_cols), 1, figsize=(10, 3*len(self.metric_cols)))
        
        for i, metric in enumerate(self.metric_cols):
            sns.histplot(df[metric], kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution of {metric}")
            
        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch}_distributions.png", dpi=300)
        plt.close()
        
        return df
        
    def analyze_metric_evolution(self):
        """Analyze how metrics evolve over epochs."""
        epochs = sorted(self.epoch_metrics.keys())
        
        if len(epochs) <= 1:
            print("Need at least 2 epochs for evolution analysis")
            return
            
        # Track metric statistics over time
        stats = {
            metric: {
                "mean": [], "median": [], "std": [], "min": [], "max": []
            } for metric in self.metric_cols
        }
        
        for epoch in epochs:
            df = self.epoch_metrics[epoch]
            
            for metric in self.metric_cols:
                stats[metric]["mean"].append(df[metric].mean())
                stats[metric]["median"].append(df[metric].median())
                stats[metric]["std"].append(df[metric].std())
                stats[metric]["min"].append(df[metric].min())
                stats[metric]["max"].append(df[metric].max())
        
        # Plot evolution of each metric
        for metric in self.metric_cols:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, stats[metric]["mean"], 'b-', label="Mean")
            plt.fill_between(
                epochs, 
                [m - s for m, s in zip(stats[metric]["mean"], stats[metric]["std"])],
                [m + s for m, s in zip(stats[metric]["mean"], stats[metric]["std"])],
                alpha=0.3
            )
            plt.plot(epochs, stats[metric]["median"], 'r--', label="Median")
            plt.title(f"Evolution of {metric}")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / f"{metric}_evolution.png", dpi=300)
            plt.close()
        
        # Compute rank stability
        if len(epochs) > 1:
            stability_data = []
            
            for i in range(len(epochs)-1):
                epoch1 = epochs[i]
                epoch2 = epochs[i+1]
                
                df1 = self.epoch_metrics[epoch1]
                df2 = self.epoch_metrics[epoch2]
                
                # Merge on sample_id
                merged = pd.merge(
                    df1[["sample_id"] + self.metric_cols],
                    df2[["sample_id"] + self.metric_cols],
                    on="sample_id",
                    suffixes=('_1', '_2')
                )
                
                for metric in self.metric_cols:
                    # Compute rank correlation
                    rho, p = spearmanr(merged[f"{metric}_1"], merged[f"{metric}_2"])
                    
                    stability_data.append({
                        "epoch_from": epoch1,
                        "epoch_to": epoch2,
                        "metric": metric,
                        "rank_correlation": rho,
                        "p_value": p
                    })
            
            stability_df = pd.DataFrame(stability_data)
            stability_df.to_csv(self.output_dir / "metric_stability.csv", index=False)
            
            # Plot stability
            for metric in self.metric_cols:
                metric_stability = stability_df[stability_df["metric"] == metric]
                
                plt.figure(figsize=(10, 6))
                plt.plot(
                    metric_stability["epoch_from"], 
                    metric_stability["rank_correlation"],
                    'o-'
                )
                plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
                plt.title(f"Rank Stability of {metric}")
                plt.xlabel("From Epoch")
                plt.ylabel("Rank Correlation")
                plt.ylim(0, 1.05)
                plt.grid(True, alpha=0.3)
                plt.savefig(self.output_dir / f"{metric}_stability.png", dpi=300)
                plt.close()
        
    def generate_curriculum(self, epoch: int = -1):
        """Generate curriculum data based on metrics.
        
        Args:
            epoch: Epoch to use (-1 for latest)
        """
        if epoch == -1:
            epoch = max(self.epoch_metrics.keys())
            
        print(f"Generating curriculum from epoch {epoch}")
        
        df = self.epoch_metrics[epoch]
        
        # Add quartiles if not present
        for metric in self.metric_cols:
            quartile_col = f"{metric}_quartile"
            if quartile_col not in df.columns:
                df[quartile_col] = pd.qcut(df[metric], 4, labels=False)
        
        # Create curriculum data
        curriculum = {}
        for _, row in df.iterrows():
            sample_id = int(row["sample_id"])
            curriculum[sample_id] = {
                f"{metric}_quartile": int(row[f"{metric}_quartile"])
                for metric in self.metric_cols
            }
        
        # Save curriculum
        with open(self.output_dir / "curriculum_data.json", "w") as f:
            json.dump(curriculum, f, indent=2)
            
        print(f"Generated curriculum data for {len(curriculum)} samples")
        
        return curriculum


    def analyze_correlations_over_time(
        self,
        reference_metric: str = "latent_error",
        rolling_window: int = 1,
    ):
        """
        Plot how the correlation between the reference metric and all
        other difficulty metrics changes over epochs.

        Args
        ----
        reference_metric : str
            The metric whose correlation with every other metric you
            want to track (default: "latent_error").
        rolling_window : int
            Optionally smooth the curves with a centred running mean
            (set to 1 to disable).
        """
        if reference_metric not in self.metric_cols:
            raise ValueError(
                f"{reference_metric!r} not in available metrics: {self.metric_cols}"
            )

        epochs = sorted(self.epoch_metrics.keys())
        if len(epochs) <= 1:
            print("Need at least 2 epochs for correlation-over-time analysis.")
            return

        # Collect correlations in a dict of lists keyed by metric name
        corr_traces: Dict[str, List[float]] = {
            m: [] for m in self.metric_cols if m != reference_metric
        }

        for epoch in epochs:
            df = self.epoch_metrics[epoch]
            for m in corr_traces.keys():
                rho, _ = spearmanr(df[reference_metric], df[m])
                corr_traces[m].append(rho)

        # Optional rolling mean for smoother curves
        if rolling_window > 1:
            from pandas import Series
            for m, vals in corr_traces.items():
                corr_traces[m] = (
                    Series(vals)
                    .rolling(window=rolling_window, center=True, min_periods=1)
                    .mean()
                    .tolist()
                )

        # ----------  plotting  ----------
        plt.figure(figsize=(10, 6))
        for m, vals in corr_traces.items():
            plt.plot(epochs, vals, marker="o", label=m)

        plt.axhline(0, color="k", linewidth=0.8, alpha=0.5)
        plt.ylim(-1.05, 1.05)
        plt.xlabel("Epoch")
        plt.ylabel(f"Spearman ρ with {reference_metric}")
        plt.title(f"Correlation of metrics with '{reference_metric}' over time")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_file = self.output_dir / f"{reference_metric}_correlation_over_time.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"[saved] {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze difficulty metrics")
    parser.add_argument("--metrics-dir", type=str, required=True, help="Directory with metrics files")
    parser.add_argument("--output-dir", type=str, help="Output directory for analysis")
    
    args = parser.parse_args()
    
    analyzer = MetricsAnalyzer(args.metrics_dir, args.output_dir)
    
    # Analyze first epoch (baseline)
    analyzer.analyze_single_epoch(0)
    
    # Analyze evolution
    analyzer.analyze_metric_evolution()
    
    # Generate curriculum
    analyzer.generate_curriculum()

    # Use 'reconstruction_error' as reference metric if available, otherwise use first available metric
    if analyzer.metric_cols:
        reference_metric = "reconstruction_error" if "reconstruction_error" in analyzer.metric_cols else analyzer.metric_cols[0]
        analyzer.analyze_correlations_over_time(reference_metric=reference_metric)
    
if __name__ == "__main__":
    main() 