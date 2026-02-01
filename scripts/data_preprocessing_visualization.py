# -*- coding: utf-8 -*-
"""
数据预处理可视化脚本
生成用于数据预处理说明文档的美观图表。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from style_config import set_style, COLORS, save_fig
except ImportError:
    # Fallback
    COLORS = {"blue": "#4c72b0", "grid": "#CCCCCC"}
    def set_style():
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = ["sans-serif"]
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False
    def save_fig(fig, path):
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

# ============================
# 配置与路径
# ============================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
DATA_FILE = DATA_DIR / "2026_MCM_Problem_C_Data.csv"

# Apply style
set_style()

# ============================
# 辅助函数
# ============================

def parse_week_number(col_name: str) -> int:
    return int(col_name.split("_")[0].replace("week", ""))

def identify_week_columns(columns):
    week_cols = [c for c in columns if c.startswith("week") and c.endswith("_score")]
    weeks = sorted({parse_week_number(c) for c in week_cols})
    return weeks

def compute_week_scores(df: pd.DataFrame, weeks):
    df = df.copy()
    for w in weeks:
        judge_cols = [c for c in df.columns if c.startswith(f"week{w}_") and c.endswith("_score")]
        scores = df[judge_cols].replace("N/A", np.nan).apply(pd.to_numeric, errors='coerce')
        m = scores.notna().sum(axis=1)
        s = scores.sum(axis=1, skipna=True)
        df[f"week{w}_avg"] = (s / m).where((m > 0) & (s > 0), np.nan)
        df[f"week{w}_std"] = scores.std(axis=1, skipna=True)
    return df

def extract_elimination_week(row):
    res = str(row.get("results", ""))
    match = re.search(r"(Eliminated|Withdrew).*Week\s+(\d+)", res, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return np.nan

def last_active_week(row, weeks):
    active_weeks = [w for w in weeks if pd.notna(row.get(f"week{w}_avg", np.nan))]
    return max(active_weeks) if active_weeks else 0

# ============================
# 原有绘图逻辑
# ============================

def plot_score_distribution_by_season(df):
    avg_cols = [c for c in df.columns if "_avg" in c]
    melted = df.melt(id_vars=["season"], value_vars=avg_cols, value_name="Average Score")
    melted = melted.dropna(subset=["Average Score"])
    plt.figure(figsize=(14, 6))
    sns.boxplot(x="season", y="Average Score", data=melted, showfliers=False, color="#4c72b0", boxprops=dict(alpha=0.6))
    plt.title("各赛季评委评分分布 (Distribution of Judge Scores by Season)", fontsize=14, pad=15)
    plt.xlabel("Season", fontsize=12)
    plt.ylabel("Judge Average Score", fontsize=12)
    plt.savefig(FIG_DIR / "eda_score_distribution_season.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_score_progression(df, weeks):
    records = []
    for idx, row in df.iterrows():
        s = row["season"]
        for w in weeks:
            col = f"week{w}_avg"
            if col in row and pd.notna(row[col]):
                records.append({"Season": s, "Week": w, "Score": row[col]})
    data_long = pd.DataFrame(records)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_long, x="Week", y="Score", errorbar="sd", marker="o", markersize=6)
    plt.title("各周次平均评分趋势 (Score Progression Over Weeks)", fontsize=14, pad=15)
    plt.xlabel("Week Number", fontsize=12)
    plt.ylabel("Average Score", fontsize=12)
    plt.savefig(FIG_DIR / "eda_score_progression.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_active_heatmap(df, weeks):
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
    plt.figure(figsize=(12, 8))
    sns.heatmap(active_matrix, cmap="YlGnBu", cbar_kws={'label': 'Active Contestants'}, annot=True, fmt="d", linewidths=.5)
    plt.title("各赛季每周参赛人数热力图", fontsize=14, pad=15)
    plt.xlabel("Week", fontsize=12)
    plt.ylabel("Season", fontsize=12)
    plt.savefig(FIG_DIR / "eda_active_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_elimination_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="last_active_week", bins=range(1, 16), kde=False, color="#d62728", alpha=0.7)
    plt.title("选手最后参赛周次分布", fontsize=14, pad=15)
    plt.xlabel("Last Active Week", fontsize=12)
    plt.xticks(range(1, 16))
    plt.savefig(FIG_DIR / "eda_elimination_dist.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_score_vs_placement_scatter(df):
    avg_cols = [c for c in df.columns if "_avg" in c]
    df["season_avg_score"] = df[avg_cols].mean(axis=1)
    if "placement" not in df.columns: return
    df["placement_num"] = pd.to_numeric(df["placement"], errors='coerce')
    clean_df = df.dropna(subset=["season_avg_score", "placement_num"])
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(x=clean_df["placement_num"], y=clean_df["season_avg_score"], 
                     c=clean_df["season"], cmap="viridis", alpha=0.6, s=50)
    plt.colorbar(sc, label="Season")
    plt.title("平均得分 vs 最终排名", fontsize=14, pad=15)
    plt.xlabel("Final Placement (1 = Winner)", fontsize=12)
    plt.ylabel("Season Average Score", fontsize=12)
    plt.xlim(0, max(clean_df["placement_num"])+1)
    plt.savefig(FIG_DIR / "eda_score_vs_placement.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_judge_disagreement_scatter(df, weeks):
    records = []
    for _, row in df.iterrows():
        for w in weeks:
            avg_col = f"week{w}_avg"
            std_col = f"week{w}_std"
            if std_col in row and pd.notna(row[avg_col]) and pd.notna(row[std_col]):
                records.append({"Mean Score": row[avg_col], "Disagreement (Std)": row[std_col], "Season": row["season"]})
    plot_data = pd.DataFrame(records)
    if plot_data.empty: return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_data, x="Mean Score", y="Disagreement (Std)", hue="Season", alpha=0.3, palette="Spectral", legend=False, s=20)
    plt.title("评委均分 vs 评分争议度", fontsize=14, pad=15)
    plt.xlabel("Average Score", fontsize=12)
    plt.ylabel("Standard Deviation", fontsize=12)
    plt.savefig(FIG_DIR / "eda_judge_disagreement.png", dpi=300, bbox_inches="tight")
    plt.close()

# ============================
# 新增的分布图表 (更多维度)
# ============================

def plot_overall_score_density(df):
    """图7: 整体评分密度分布 (Split by Early vs Late seasons)"""
    avg_cols = [c for c in df.columns if "_avg" in c]
    melted = df.melt(id_vars=["season"], value_vars=avg_cols, value_name="Average Score")
    melted = melted.dropna(subset=["Average Score"])
    
    # 将赛季分为 "Early (1-15)" 和 "Late (16+)"
    melted["Era"] = melted["season"].apply(lambda s: "Early Seasons (1-15)" if s <= 15 else "Late Seasons (16+)")
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=melted, x="Average Score", hue="Era", fill=True, common_norm=False, alpha=0.4)
    plt.title("评分密度分布：早期赛季 vs 晚期赛季", fontsize=14, pad=15)
    plt.xlabel("Average Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
    plt.savefig(FIG_DIR / "eda_score_density_era.png", dpi=300, bbox_inches="tight")
    print(f"Generated: {FIG_DIR / 'eda_score_density_era.png'}")
    plt.close()

def plot_survival_curve(df, weeks):
    """图8: 每一周的选手留存率 (Survival Rate)"""
    season_counts = df.groupby("season").size()
    survival_data = []
    
    for s in df["season"].unique():
        s_df = df[df["season"] == s]
        total = len(s_df)
        if total == 0: continue
        for w in weeks:
            active = s_df[f"week{w}_avg"].notna().sum()
            rate = active / total
            survival_data.append({"Season": s, "Week": w, "Survival Rate": rate})
            
    plot_df = pd.DataFrame(survival_data)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x="Week", y="Survival Rate", errorbar="sd", color="purple")
    # 也可以画每条线
    # sns.lineplot(data=plot_df, x="Week", y="Survival Rate", units="Season", estimator=None, alpha=0.1, color="grey")
    
    plt.title("选手留存率随时间变化 (Survival Rate Over Weeks)", fontsize=14, pad=15)
    plt.xlabel("Week", fontsize=12)
    plt.ylabel("Proportion of Contestants Remaining", fontsize=12)
    plt.ylim(0, 1.05)
    
    plt.savefig(FIG_DIR / "eda_survival_curve.png", dpi=300, bbox_inches="tight")
    print(f"Generated: {FIG_DIR / 'eda_survival_curve.png'}")
    plt.close()

def plot_score_volatility(df, weeks):
    """图9: 分数改进分布 (Week-over-Week Change)"""
    # 只有连续两周都有分的才算
    deltas = []
    for idx, row in df.iterrows():
        # sorted weeks
        for i in range(len(weeks) - 1):
            w_curr = weeks[i]
            w_next = weeks[i+1]
            # check consecutive? Assuming weeks list is continuous integers usually
            # But let's just diff any adjacent available columns
            col_curr = f"week{w_curr}_avg"
            col_next = f"week{w_next}_avg"
            
            if pd.notna(row.get(col_curr)) and pd.notna(row.get(col_next)):
                delta = row[col_next] - row[col_curr]
                deltas.append(delta)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(deltas, bins=50, kde=True, color="teal")
    plt.axvline(0, color="k", linestyle="--", alpha=0.5)
    
    plt.title("选手每周分数变化分布 (Week-over-Week Score Improvement)", fontsize=14, pad=15)
    plt.xlabel("Score Change (Next Week - Current Week)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    plt.savefig(FIG_DIR / "eda_score_volatility.png", dpi=300, bbox_inches="tight")
    print(f"Generated: {FIG_DIR / 'eda_score_volatility.png'}")
    plt.close()

def plot_feature_importance_like_reference(df, weeks):
    """
    Generates a horizontal bar chart of feature importance for predicting Placement.
    Mimics the style of the user-provided reference image.
    Features: Age, Season, Avg Score, Score Volatility, Start Score, etc.
    """
    if not SKLEARN_AVAILABLE:
        print("Skipping Feature Importance plot (scikit-learn not found).")
        return

    # 1. Feature Engineering
    # We want to predict 'placement' (numerical). 
    # If placement is non-numeric (e.g., 'Winner'), it usually is parsed to 1 in previous steps.
    
    # Ensure placement is numeric
    if "placement" not in df.columns:
        return
        
    df_work = df.copy()
    df_work["placement_num"] = pd.to_numeric(df_work["placement"], errors='coerce')
    
    # Calculate aggregated features
    avg_cols = [f"week{w}_avg" for w in weeks if f"week{w}_avg" in df_work.columns]
    
    if not avg_cols:
        return

    # Row-wise statistics
    df_work["Avg_Score"] = df_work[avg_cols].mean(axis=1)
    df_work["Max_Score"] = df_work[avg_cols].max(axis=1)
    df_work["Min_Score"] = df_work[avg_cols].min(axis=1)
    df_work["Score_Std"] = df_work[avg_cols].std(axis=1).fillna(0)
    
    # Early performance (Week 1 or first available)
    if f"week{weeks[0]}_avg" in df_work.columns:
        df_work["Start_Week_Score"] = df_work[f"week{weeks[0]}_avg"]
    else:
        df_work["Start_Week_Score"] = df_work["Avg_Score"] # Fallback

    # Feature selection
    feature_cols = [
        "season", 
        "celebrity_age_during_season", 
        "Avg_Score", 
        "Max_Score", 
        "Min_Score", 
        "Score_Std",
        "Start_Week_Score"
    ]
    
    # Clean data for modeling
    model_df = df_work[feature_cols + ["placement_num"]].dropna()
    
    if model_df.empty:
        return

    X = model_df[feature_cols]
    y = model_df["placement_num"]

    # 2. Train Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    
    # 3. Visualization
    # Scale to be "integer-like" scores if preferred, or just percentage.
    # The reference image has values like 573, 522. Assuming sum=1000 scale.
    scaled_importances = importances * 1000
    
    # Create DataFrame for plotting
    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": scaled_importances
    }).sort_values("Importance", ascending=True) # Ascending for barh (bottom to top)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Horizontal bars
    # Use thin bars as in reference image
    y_pos = np.arange(len(feat_imp))
    ax.barh(y_pos, feat_imp["Importance"], height=0.3, color=COLORS.get("blue", "#4c72b0"))
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_imp["Feature"])
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Features")
    ax.set_title("Feature importance")
    
    # Comparison style: Grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.9)
    ax.grid(True, axis='y', linestyle='-', alpha=0.9)
    # The reference has full grid
    
    # Add values at the end of bars
    for i, v in enumerate(feat_imp["Importance"]):
        ax.text(v + 10, i, f"{int(v)}", va='center', fontweight='normal', fontsize=10)
    
    # Adjust x limit to fit labels
    ax.set_xlim(0, max(scaled_importances) * 1.15)
    
    save_fig(fig, FIG_DIR / "data_preprocessing_feature_importance.png")
    print(f"Generated: {FIG_DIR / 'data_preprocessing_feature_importance.png'}")


def main():
    if not DATA_FILE.exists(): return
    print("Loading...")
    df = pd.read_csv(DATA_FILE)
    weeks = identify_week_columns(df.columns)
    df = compute_week_scores(df, weeks)
    
    df["elimination_week_regex"] = df.apply(extract_elimination_week, axis=1)
    df["last_active_week"] = df.apply(lambda r: last_active_week(r, weeks), axis=1) # Simplified for viz

    print("Generating all plots...")
    plot_overall_score_density(df)
    plot_survival_curve(df, weeks)
    plot_score_volatility(df, weeks)
    
    # Run other plots
    plot_score_vs_placement_scatter(df)
    plot_judge_disagreement_scatter(df, weeks)
    plot_score_distribution_by_season(df)
    plot_score_progression(df, weeks)
    plot_active_heatmap(df, weeks)
    plot_elimination_distribution(df)

    # New feature importance plot
    print("Generating feature importance plot...")
    plot_feature_importance_like_reference(df, weeks)
    
    print("Done.")

if __name__ == "__main__":
    main()
