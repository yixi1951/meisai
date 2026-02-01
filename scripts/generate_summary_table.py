import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_FILE = ROOT_DIR / "data" / "2026_MCM_Problem_C_Data.csv"
    VOTES_FILE = ROOT_DIR / "outputs" / "fan_vote_estimates_q1a.csv"
    OUTPUT_FILE = ROOT_DIR / "outputs" / "contestant_weekly_fan_votes_summary.csv"
    
    # Load Data
    print("Loading data...")
    raw_df = pd.read_csv(DATA_FILE)
    votes_df = pd.read_csv(VOTES_FILE)
    
    # Select static columns from raw data
    static_cols = ["season", "celebrity_name", "ballroom_partner", "placement", "results"]
    contestant_info = raw_df[static_cols].copy()
    
    # Merge
    # Note: celebrity_name should be unique within a season usually, but let's be safe with season key
    print("Merging data...")
    merged_df = pd.merge(votes_df, contestant_info, on=["season", "celebrity_name"], how="left")
    
    # Generate contestant_row (index within season)
    # Get unique contestants per season and assign an ID
    contestants = merged_df[["season", "celebrity_name"]].drop_duplicates().sort_values(["season", "celebrity_name"])
    contestants["contestant_row"] = contestants.groupby("season").cumcount()
    
    merged_df = pd.merge(merged_df, contestants, on=["season", "celebrity_name"], how="left")
    
    # Calculate Judge Share Q (if not already strictly present or needs recalculation to match total=1)
    # The file has judge_total. Let's compute share per week.
    print("Calculating shares...")
    week_stats = merged_df.groupby(["season", "week"])["judge_total"].sum().reset_index().rename(columns={"judge_total": "week_judge_sum"})
    merged_df = pd.merge(merged_df, week_stats, on=["season", "week"], how="left")
    
    merged_df["judge_share_Q"] = merged_df["judge_total"] / merged_df["week_judge_sum"]
    
    # Calculate Composite C
    # Based on the user image: C = Judge Share + Fan Share
    merged_df["composite_C"] = merged_df["judge_share_Q"] + merged_df["fan_vote_share"]
    
    # Rename and Select Columns
    final_cols = {
        "season": "season",
        "week": "week",
        "contestant_row": "contestant_row",
        "celebrity_name": "celebrity_name",
        "ballroom_partner": "ballroom_partner",
        "placement": "placement",
        "results": "results",
        "judge_total": "judge_total_S",
        "judge_share_Q": "judge_share_Q",
        "fan_vote_share": "fan_share_P",
        "fan_votes_absolute": "fan_votes_V",
        "composite_C": "composite_C"
    }
    
    output_df = merged_df.rename(columns=final_cols)[list(final_cols.values())]
    
    # Format and Sort
    output_df = output_df.sort_values(["season", "week", "composite_C"], ascending=[True, True, False])
    
    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    # Display sample similarly to image (Season 1, first few weeks)
    sample = output_df[output_df["season"] == 1].head(15)
    print("\nSample Output (Season 1):")
    print(sample.to_string())

if __name__ == "__main__":
    main()
