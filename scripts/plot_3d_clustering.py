import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path
import seaborn as sns

# Set style for academic plot
sns.set_theme(style="white")

def main():
    # File Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_FILE = ROOT_DIR / "outputs" / "problem3_model_dataset.csv"
    FIG_DIR = ROOT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # 2. Prepare Data for 3D Clustering
    # We want to cluster contestants based on their performance characteristics.
    # Let's aggregate by contestant first to get their "overall profile".
    contestant_stats = df.groupby(["celebrity_name", "season"]).agg({
        "judge_total": ["mean", "std"],     # Average Score & Consistency
        "fan_vote_share": "mean"            # Popularity
    }).reset_index()
    
    # Flatten columns
    contestant_stats.columns = ["celebrity_name", "season", "avg_judge_score", "score_volatility", "avg_fan_share"]
    contestant_stats["score_volatility"] = contestant_stats["score_volatility"].fillna(0) # Handle single-week contestants
    
    # Features for Clustering: 
    # X: Avg Judge Score (Technical Skill)
    # Y: Score Volatility (Consistency)
    # Z: Avg Fan Share (Popularity)
    features = ["avg_judge_score", "score_volatility", "avg_fan_share"]
    X = contestant_stats[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. K-Means Clustering
    k = 3 # We want 3 groups like the image (e.g., "Frontrunners", "Controversial", "Early Exits")
    kmeans = KMeans(n_clusters=k, random_state=42)
    contestant_stats["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Map clusters to meaningful labels (heuristic)
    # Calculate mean judge score per cluster to identify them
    cluster_means = contestant_stats.groupby("cluster")["avg_judge_score"].mean().sort_values()
    
    # Assign labels based on score rank (Low -> Medium -> High)
    label_map = {
        cluster_means.index[0]: "Underdogs (Low Score)",
        cluster_means.index[1]: "Mid-Tier (Average)",
        cluster_means.index[2]: "Frontrunners (High Score)"
    }
    contestant_stats["cluster_label"] = contestant_stats["cluster"].map(label_map)
    
    # 4. Generate 3D Scatter Plot
    print("Generating 3D Scatter Plot...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors
    colors = {
        "Underdogs (Low Score)": "#2ca02c",   # Green (Easy/Low like in image)
        "Mid-Tier (Average)": "#d62728",      # Red (Middle)
        "Frontrunners (High Score)": "#9467bd" # Purple (Difficult/High)
    }
    
    for label, color in colors.items():
        subset = contestant_stats[contestant_stats["cluster_label"] == label]
        ax.scatter(
            subset["avg_judge_score"], 
            subset["score_volatility"], 
            subset["avg_fan_share"],
            c=color,
            label=label,
            s=40,
            alpha=0.7,
            edgecolors='w'
        )
    
    # Labels and Formatting
    ax.set_xlabel('Avg Judge Score (Skill)')
    ax.set_ylabel('Score Volatility (Inconsistency)')
    ax.set_zlabel('Avg Fan Share (Popularity)')
    ax.set_title('3D Clustering of Contestant Performance Profiles', fontsize=14)
    
    # Match the "Planar Panel" look of the user image (the grid walls)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True)
    
    ax.legend(title="Contestant Type", loc='upper right')
    
    out_path = FIG_DIR / "contestant_clustering_3d.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved 3D plot to {out_path}")

if __name__ == "__main__":
    main()
