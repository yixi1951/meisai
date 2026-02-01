import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"] # Support Chinese if needed
plt.rcParams["axes.unicode_minus"] = False

def main():
    # Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    INPUT_FILE = ROOT_DIR / "outputs" / "contestant_weekly_fan_votes_summary.csv"
    FIG_DIR = ROOT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # ---------------------------------------------------------
    # Plot 1: Scatter of Judge Share vs Fan Share (Bubbles)
    # Colored by Final Placement grouping
    # ---------------------------------------------------------
    print("Generating Judge vs Fan Share Scatter Plot...")
    
    # Create categories for placement
    def categorize_placement(p):
        if p == 1: return "Winner (1st)"
        elif p <= 3: return "Finalist (2nd-3rd)"
        elif p <= 6: return "Top Tier (4th-6th)"
        else: return "Eliminated Early (>6th)"
        
    df["Placement Group"] = df["placement"].apply(categorize_placement)
    
    # Sort order for legend
    hue_order = ["Winner (1st)", "Finalist (2nd-3rd)", "Top Tier (4th-6th)", "Eliminated Early (>6th)"]
    palette = {"Winner (1st)": "#d62728", "Finalist (2nd-3rd)": "#ff7f0e", 
               "Top Tier (4th-6th)": "#2ca02c", "Eliminated Early (>6th)": "#7f7f7f"}
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="judge_share_Q",
        y="fan_share_P",
        hue="Placement Group",
        hue_order=hue_order,
        palette=palette,
        size="placement", # Just to add some variation, or maybe ignore size
        sizes=(20, 100),
        alpha=0.6,
        edgecolor=None
    )
    
    # Add diagonal line (Perfect Agreement)
    limit = max(df["judge_share_Q"].max(), df["fan_share_P"].max())
    plt.plot([0, limit], [0, limit], color="gray", linestyle="--", alpha=0.5, label="Perfect Agreement")
    
    plt.title("Judge Share vs Fan Share Correlation\n(Colored by Season Final Placement)", fontsize=14)
    plt.xlabel("Judge Share (Q)", fontsize=12)
    plt.ylabel("Fan Vote Share (P)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path_1 = FIG_DIR / "summary_scatter_judge_vs_fan.png"
    plt.savefig(out_path_1, dpi=300)
    print(f"Saved {out_path_1}")
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Trajectory of Composite Score for a Specific Season
    # Let's pick a season with interesting data, e.g., Season 17 or 27 (from previous context)
    # ---------------------------------------------------------
    target_season = 19
    print(f"Generating Composite Score Trajectory for Season {target_season}...")
    
    season_df = df[df["season"] == target_season].copy()
    
    if not season_df.empty:
        # Get top 4 finalists to reduce clutter
        top_contestants = season_df.sort_values("placement")["celebrity_name"].unique()[:5]
        season_df = season_df[season_df["celebrity_name"].isin(top_contestants)]
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=season_df,
            x="week",
            y="composite_C",
            hue="celebrity_name",
            marker="o",
            linewidth=2.5,
            palette="tab10"
        )
        
        plt.title(f"Composite Score Trajectory (Season {target_season} Top 5)", fontsize=14)
        plt.xlabel("Week", fontsize=12)
        plt.ylabel("Composite Score C = Q + P", fontsize=12)
        plt.legend(title="Contestant", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        out_path_2 = FIG_DIR / f"summary_trajectory_season_{target_season}.png"
        plt.savefig(out_path_2, dpi=300)
        print(f"Saved {out_path_2}")
        plt.close()
    else:
        print(f"No data for season {target_season}")

    # ---------------------------------------------------------
    # Plot 3: Stacked Composition (Judge vs Fan) for Season 1 Winners
    # ---------------------------------------------------------
    print("Generating Stacked Composition for Season 1 Winners...")
    s1_df = df[(df["season"] == 1) & (df["placement"] <= 3)].copy()
    
    # Average shares across all weeks for simplicity in this view
    s1_avg = s1_df.groupby(["celebrity_name", "placement"])[["judge_share_Q", "fan_share_P"]].mean().reset_index()
    s1_avg = s1_avg.sort_values("placement")
    
    # Melt for stacked bar
    s1_melted = s1_avg.melt(id_vars=["celebrity_name", "placement"], value_vars=["judge_share_Q", "fan_share_P"], var_name="Component", value_name="Average Share")
    
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=s1_melted,
        x="celebrity_name",
        y="Average Share",
        hue="Component",
        palette={"judge_share_Q": "#1f77b4", "fan_share_P": "#ff7f0e"}
    )
    
    plt.title("Season 1 Top 3: Average Score Composition", fontsize=14)
    plt.xlabel("Contestant (Ranked 1st to 3rd)", fontsize=12)
    plt.ylabel("Average Share", fontsize=12)
    plt.tight_layout()
    
    out_path_3 = FIG_DIR / "summary_stacked_composition_s1.png"
    plt.savefig(out_path_3, dpi=300)
    print(f"Saved {out_path_3}")
    plt.close()

if __name__ == "__main__":
    main()
