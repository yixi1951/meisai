import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Set style to match the academic/professional look
sns.set_theme(style="ticks")

def create_hexbin_comparison(data_df, x_col, y_col, color, output_path, title=None):
    """
    Creates a hexbin jointplot with marginal histograms.
    
    Args:
        data_df (pd.DataFrame): Dataframe containing the data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        color (str): Hex color code for the plot.
        output_path (Path): Path to save the figure.
        title (str, optional): Title for the plot.
    """
    
    # Create the jointplot
    g = sns.jointplot(
        data=data_df,
        x=x_col,
        y=y_col,
        kind="hex",
        color=color,
        height=8,
        ratio=5,
        space=0.1,
        # Remove hardcoded limits to allow auto-scaling for different data ranges
        # xlim=(3.0, 6.0),
        # ylim=(3.0, 6.0)
    )
    
    # Adjust labels
    g.set_axis_labels(x_col, y_col, fontsize=12)
    
    if title:
        g.fig.suptitle(title, y=1.02, fontsize=14)
        
    # Save
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to {output_path}")
    plt.close()

def main():
    # Define output directory
    FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
    INPUT_FILE = Path(__file__).resolve().parent.parent / "outputs" / "fair_rule_weekly_q4.csv"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading real data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Data file not found. Please ensure 'outputs/fair_rule_weekly_q4.csv' exists.")
        return

    # Colors extracted roughly from the image
    ORANGE_COLOR = "#E69F00" # Approximate gold/orange
    BLUE_COLOR = "#56B4E9"   # Approximate specific blue

    print("Creating Orange Chart (Judge Share vs Fan Share)...")
    # Setting limits based on data distribution (0 to ~0.5 covers 99%)
    create_hexbin_comparison(
        df, 
        "judge_share", 
        "fan_share", 
        ORANGE_COLOR, 
        FIG_DIR / "hexbin_judge_vs_fan.png",
        title="Judge Share vs. Fan Vote Share"
    )
    
    print("Creating Blue Chart (Percent Score vs Fair Score)...")
    # Setting limits based on data distribution
    create_hexbin_comparison(
        df, 
        "percent_score", 
        "fair_score", 
        BLUE_COLOR, 
        FIG_DIR / "hexbin_percent_vs_fair.png",
        title="Standard Percent Score vs. Fair Rule Score"
    )

if __name__ == "__main__":
    main()
