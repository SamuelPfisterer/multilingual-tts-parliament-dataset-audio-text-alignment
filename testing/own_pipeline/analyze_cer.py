import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np
import os

def load_alignments(json_path: str) -> pd.DataFrame:
    """Load aligned segments from JSON file into DataFrame.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        DataFrame with alignment data
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["segments"])

def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute various statistics about CER distribution.
    
    Args:
        df: DataFrame containing alignment data
        
    Returns:
        Dictionary with computed statistics
    """
    # Define CER thresholds for analysis
    cer_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3]
    
    # Basic statistics - convert numpy types to native Python types
    stats = {
        "mean_cer": float(df["cer"].mean()),
        "median_cer": float(df["cer"].median()),
        "std_cer": float(df["cer"].std()),
        "min_cer": float(df["cer"].min()),
        "max_cer": float(df["cer"].max()),
        "total_segments": int(len(df)),
        "percentile_10": float(df["cer"].quantile(0.1)),
        "percentile_25": float(df["cer"].quantile(0.25)),
        "percentile_75": float(df["cer"].quantile(0.75)),
        "percentile_90": float(df["cer"].quantile(0.9))
    }
    
    # Add threshold statistics - convert numpy types to native Python types
    for threshold in cer_thresholds:
        stats[f"segments_below_{threshold}"] = int((df["cer"] < threshold).sum())
        stats[f"percent_below_{threshold}"] = float((stats[f"segments_below_{threshold}"] / stats["total_segments"]) * 100)
    
    return stats

def analyze_cer_jumps(df: pd.DataFrame, output_dir: str, threshold_jump: float = 0.3, context_window: int = 2):
    """Analyze segments around significant CER jumps.
    
    Args:
        df: DataFrame containing alignment data
        output_dir: Directory to save results
        threshold_jump: Minimum CER difference to consider as a jump
        context_window: Number of segments to show before and after the jump
    """
    # Calculate CER differences between consecutive segments
    cer_diffs = df["cer"].diff()
    
    # Find indices where jumps occur
    jump_indices = cer_diffs[abs(cer_diffs) > threshold_jump].index
    
    # Prepare jumps data for JSON
    jumps_data = []
    
    if len(jump_indices) == 0:
        print("\nNo significant CER jumps found.")
    else:
        print(f"\nFound {len(jump_indices)} significant CER jumps (threshold: {threshold_jump}):")
        
        for jump_idx in jump_indices:
            # Get context window indices
            start_idx = max(0, jump_idx - context_window)
            end_idx = min(len(df), jump_idx + context_window + 1)
            
            # Collect context segments
            context_segments = []
            context_df = df.iloc[start_idx:end_idx]
            for idx, row in context_df.iterrows():
                segment_data = {
                    "segment_idx": idx,
                    "is_jump_point": idx == jump_idx,
                    "start_time": row["start"],
                    "end_time": row["end"],
                    "cer": row["cer"],
                    "asr_text": row["asr_text"],
                    "human_text": row["human_text"]
                }
                context_segments.append(segment_data)
            
            jump_data = {
                "jump_index": jump_idx,
                "cer_change": cer_diffs[jump_idx],
                "context_segments": context_segments
            }
            jumps_data.append(jump_data)
            
            # Print information
            print(f"\n{'='*80}")
            print(f"Jump at segment {jump_idx} (CER change: {cer_diffs[jump_idx]:.3f})")
            print(f"{'='*80}")
            
            for segment in context_segments:
                marker = ">>> " if segment["is_jump_point"] else "    "
                print(f"\n{marker}Segment {segment['segment_idx']}:")
                print(f"    Time: {segment['start_time']:.1f}s → {segment['end_time']:.1f}s")
                print(f"    CER: {segment['cer']:.3f}")
                print(f"    ASR Text: {segment['asr_text']}")
                print(f"    Human Text: {segment['human_text']}")
    
    # Save jumps data to JSON
    jumps_file = os.path.join(output_dir, "cer_jumps.json")
    with open(jumps_file, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": threshold_jump,
            "context_window": context_window,
            "total_jumps": len(jump_indices),
            "jumps": jumps_data
        }, f, indent=4)

def plot_cer_distribution(df: pd.DataFrame, output_dir: str):
    """Create various plots for CER analysis.
    
    Args:
        df: DataFrame containing alignment data
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use("seaborn-v0_8")
    
    # 1. Histogram with KDE (zoomed in on low CER values)
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(data=df, x="cer", kde=True, bins=50)
    plt.title("Distribution of Character Error Rates (Low CER Range)")
    plt.xlabel("Character Error Rate (CER)")
    plt.ylabel("Count")
    plt.xlim(0, 0.2)  # Focus on low CER range
    plt.savefig(f"{output_dir}/cer_distribution_low_range.png")
    plt.close()
    
    # 2. Cumulative distribution plot
    plt.figure(figsize=(10, 6))
    sorted_cer = np.sort(df["cer"])
    yvals = np.arange(len(sorted_cer)) / float(len(sorted_cer))
    plt.plot(sorted_cer, yvals)
    plt.title("Cumulative Distribution of CER")
    plt.xlabel("Character Error Rate (CER)")
    plt.ylabel("Cumulative Proportion")
    plt.grid(True)
    plt.savefig(f"{output_dir}/cer_cumulative_distribution.png")
    plt.close()
    
    # 3. Box plot (zoomed in on low CER values)
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x="cer", showfliers=False)
    plt.title("CER Box Plot (Low CER Range)")
    plt.xlabel("Character Error Rate (CER)")
    plt.xlim(0, 0.2)  # Focus on low CER range
    plt.savefig(f"{output_dir}/cer_boxplot_low_range.png")
    plt.close()
    
    # 4. CER vs Segment Duration (zoomed in on low CER values)
    plt.figure(figsize=(10, 6))
    df["duration"] = df["end"] - df["start"]
    sns.scatterplot(data=df, x="duration", y="cer")
    plt.title("CER vs Segment Duration (Low CER Range)")
    plt.xlabel("Segment Duration (seconds)")
    plt.ylabel("Character Error Rate (CER)")
    plt.ylim(0, 0.2)  # Focus on low CER range
    plt.savefig(f"{output_dir}/cer_vs_duration_low_range.png")
    plt.close()

    # 5. CER over Segment Index with annotations for big jumps
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["cer"], marker='o', linestyle='-', markersize=3, alpha=0.5)
    
    # Add annotations for big jumps
    cer_diffs = df["cer"].diff()
    jump_indices = cer_diffs[abs(cer_diffs) > 0.3].index
    
    for idx in jump_indices:
        plt.annotate(f'Jump',
                    xy=(idx, df["cer"].iloc[idx]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.title("CER over Segment Index")
    plt.xlabel("Segment Index")
    plt.ylabel("Character Error Rate (CER)")
    plt.grid(True)
    
    # Add horizontal lines for key thresholds
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.3]:
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.3)
    
    plt.savefig(f"{output_dir}/cer_over_index.png")
    plt.close()

def main():
    # Load data
    json_path = "output/alignments/7501579_3840s_aligned.json"
    df = load_alignments(json_path)
    
    # Create output directory structure
    base_dir = "cer_analyze"
    file_name = os.path.splitext(os.path.basename(json_path))[0]
    output_dir = os.path.join(base_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute statistics
    stats = compute_statistics(df)
    
    # Save statistics to JSON
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    
    # Print statistics
    print(f"\nCER Statistics (saved to {stats_file}):")
    print("-" * 50)
    print(f"Total Segments: {stats['total_segments']}")
    print(f"Mean CER: {stats['mean_cer']:.3f}")
    print(f"Median CER: {stats['median_cer']:.3f}")
    print(f"Std Dev CER: {stats['std_cer']:.3f}")
    print(f"Min CER: {stats['min_cer']:.3f}")
    print(f"Max CER: {stats['max_cer']:.3f}")
    print(f"10th Percentile: {stats['percentile_10']:.3f}")
    print(f"25th Percentile: {stats['percentile_25']:.3f}")
    print(f"75th Percentile: {stats['percentile_75']:.3f}")
    print(f"90th Percentile: {stats['percentile_90']:.3f}")
    
    print("\nSegment Quality Distribution:")
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.3]:
        print(f"Segments with CER < {threshold:.2f}: "
              f"{stats[f'segments_below_{threshold}']} "
              f"({stats[f'percent_below_{threshold}']:.1f}%)")
    
    # Create plots in the file-specific directory
    plot_cer_distribution(df, output_dir)
    print(f"\nPlots have been saved to: {output_dir}")

    # Analyze CER jumps and save to JSON
    analyze_cer_jumps(df, output_dir)

if __name__ == "__main__":
    main() 