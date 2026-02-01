
import pandas as pd
import sys
from pathlib import Path
sys.path.append('scripts')
from data_preprocessing_visualization import plot_industry_score_trend, identify_week_columns, compute_week_scores, ROOT_DIR, DATA_FILE, set_style

def main():
    set_style()
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    weeks = identify_week_columns(df.columns)
    # We essentially need raw df for this function as it calculates its own sums
    print("Running plot_industry_score_trend...")
    plot_industry_score_trend(df, weeks)
    print("Done.")

if __name__ == "__main__":
    main()
