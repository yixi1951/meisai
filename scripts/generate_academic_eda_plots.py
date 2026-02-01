
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import style_config

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
DATA_FILE = DATA_DIR / "2026_MCM_Problem_C_Data.csv"

# Apply style
style_config.set_style()

def plot_academic_active_heatmap(df, weeks):
    print("Generating Academic Active Contestants Heatmap...")
    season_week_counts = []
    all_seasons = sorted(df["season"].unique())
    for s in all_seasons:
        s_df = df[df["season"] == s]
        counts = []
        for w in weeks:
            c = s_df[f"week{w}_avg"].notna().sum()
            counts.append(c)
        season_week_counts.append(counts)
    active_matrix = pd.DataFrame(season_week_counts, index=all_seasons, columns=weeks)
    
    # Filter heatmap data to remove entirely 0 columns if any (optional)
    # But usually keeping Week 1-11 is good structure
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Academic Heatmap Style: 'YlGnBu' is good, or 'Blues' for simpler look
    # cbar_kws aspect/location optimization
    sns.heatmap(active_matrix, cmap="YlGnBu", cbar_kws={'label': 'Number of Active Contestants'}, 
                annot=True, fmt="d", linewidths=0.5, linecolor='white',
                square=False, ax=ax, annot_kws={"size": 8})
    
    # Titles and Labels (English, Standard)
    ax.set_title("Heatmap of Active Contestants by Season and Week", fontsize=14, pad=15, loc='left')
    ax.set_xlabel("Competition Week", fontsize=12)
    ax.set_ylabel("Season", fontsize=12)
    
    # Clean ticks
    ax.tick_params(axis='both', which='both', length=0)
    plt.yticks(rotation=0)
    
    out_path = FIG_DIR / "eda_active_heatmap_academic.png"
    save_fig(fig, out_path)
    print(f"Generated: {out_path}")


def plot_academic_score_distribution(df):
    print("Generating Academic Score Distribution Boxplot...")
    avg_cols = [c for c in df.columns if "_avg" in c]
    melted = df.melt(id_vars=["season"], value_vars=avg_cols, value_name="Average Score")
    melted = melted.dropna(subset=["Average Score"])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Academic Boxplot: Diverse colors (Spectral palette) for visual distinction across seasons
    # Using 'hue' set to 'season' to enable palette usage, legend=False to hide since x-axis covers it.
    sns.boxplot(x="season", y="Average Score", hue="season", data=melted, 
                showfliers=False, palette="Spectral", legend=False,
                width=0.7, linewidth=1.0,
                boxprops=dict(alpha=0.8, edgecolor='#2c3e50'),
                medianprops=dict(color="#2c3e50", linewidth=1.2),
                whiskerprops=dict(color="#2c3e50", linewidth=1.0),
                capprops=dict(color="#2c3e50", linewidth=1.0),
                ax=ax)
    
    # Clean academic styling
    ax.set_title("Longitudinal Distribution of Judge Scores Across 34 Seasons", fontsize=14, pad=15, loc='left')
    ax.set_xlabel("Season Number", fontsize=12)
    ax.set_ylabel("Judge Average Score", fontsize=12)
    
    # Grid
    ax.grid(True, axis='y', linestyle=':', color='#bfbfbf', alpha=0.6)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    out_path = FIG_DIR / "eda_score_distribution_season_academic.png"
    save_fig(fig, out_path)
    print(f"Generated: {out_path}")

def save_fig(fig, path):
    """Local save wrapper because style_config might not handle layout perfectly for these charts"""
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # Re-calculate or identify week columns as in original script
    # We need compute_week_scores logic locally or imported
    # Simple manual identification for weeks 1-11 approx
    week_cols_raw = [c for c in df.columns if c.startswith("week") and c.endswith("_score")]
    weeks = sorted({int(c.split("_")[0].replace("week", "")) for c in week_cols_raw})
    
    # Helper to compute avg as active indicator
    df_proc = df.copy()
    for w in weeks:
        judge_cols = [c for c in df.columns if c.startswith(f"week{w}_") and c.endswith("_score")]
        # Scores to numeric
        scores = df[judge_cols].replace("N/A", np.nan).apply(pd.to_numeric, errors='coerce')
        m = scores.notna().sum(axis=1)
        s = scores.sum(axis=1, skipna=True)
        df_proc[f"week{w}_avg"] = (s / m).where((m > 0) & (s > 0), np.nan)
        
    plot_academic_active_heatmap(df_proc, weeks)
    plot_academic_score_distribution(df_proc)

if __name__ == "__main__":
    main()
