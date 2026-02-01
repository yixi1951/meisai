
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import style_config

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
FIG_DIR = ROOT_DIR / "figures"
FAN_VOTE_FILE = OUTPUT_DIR / "fan_vote_estimates_q1a.csv"

# Apply style
style_config.set_style()

def plot_academic_trajectory(season=27):
    print(f"Loading data from {FAN_VOTE_FILE}...")
    fan_df = pd.read_csv(FAN_VOTE_FILE)
    
    s_df = fan_df[fan_df["season"] == season].copy()
    if s_df.empty:
        print(f"Season {season} data empty.")
        return

    # Normalize judge share
    try:
        s_df["judge_share_calc"] = s_df.groupby("week")["judge_total"].transform(lambda x: x / x.sum())
    except Exception as e:
        print(f"Error calculating judge share: {e}")
        return

    # Select Core Competitors
    target_names = ["Milo Manheim", "Alexis Ren", "Evanna Lynch", "Bobby Bones"]
    existing_targets = [name for name in target_names if name in s_df["celebrity_name"].unique()]
    
    # Setup Figure (Scientific/Nature style dimensions)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Specific colors for low saturation academic look
    # Milo (Blue-ish), Alexis (Green-ish), Evanna (Purple-ish), Bobby (Red-ish)
    color_map = {
        "Milo Manheim": "#4c72b0",   # Soft Blue
        "Alexis Ren": "#55a868",     # Soft Green
        "Evanna Lynch": "#8172b2",   # Soft Purple
        "Bobby Bones": "#c44e52"     # Soft Red
    }
    fallback_colors = sns.color_palette("deep", 10).as_hex()
    markers = ['o', 's', '^', 'D']
    
    for i, name in enumerate(existing_targets):
        c_data = s_df[s_df["celebrity_name"] == name].sort_values("week")
        
        c = color_map.get(name, fallback_colors[i % len(fallback_colors)])
        m = markers[i % len(markers)]
        
        # Plot judge share (Solid line)
        ax.plot(c_data["week"], c_data["judge_share_calc"], 
                color=c, linestyle='-', marker=m, markersize=4, linewidth=1.5, alpha=0.9,
                label=f"{name} (Judge)")
        
        # Plot fan share (Dashed line)
        ax.plot(c_data["week"], c_data["fan_vote_share"], 
                color=c, linestyle='--', marker=m, markersize=4, linewidth=1.5, alpha=0.9,
                label=f"{name} (Fan Vote)")
                
    # Academic Styling
    ax.set_title("Evolution of Share Metrics: Judges vs. Fan Votes (Season 27)", fontsize=14, loc='left', pad=15)
    ax.set_xlabel("Competition Week Week", fontsize=12)
    ax.set_ylabel("Share of Total (Proportion)", fontsize=12)
    
    # Customize Legend (Scientific Style)
    # Use 2 columns to save vertical space
    ax.legend(loc='upper left', ncol=2, frameon=False, fontsize=9, columnspacing=1.0)
    
    # Grid and Spines
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='#bfbfbf', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    out_path = FIG_DIR / f"fan_vote_trajectory_s{season}_academic.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Generated academic figure at: {out_path}")

if __name__ == "__main__":
    plot_academic_trajectory(27)
