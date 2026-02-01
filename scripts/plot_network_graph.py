import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Setup paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_FILE = ROOT_DIR / "outputs" / "problem3_model_dataset.csv"
    FIG_DIR = ROOT_DIR / "figures"
    FIG_DIR.mkdir(exist_ok=True)
    
    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("Data file not found.")
        return

    # Create a "Contestant Interaction/Similarity Network"
    # Logic: If two contestants compete in the same season, they are connected from the start.
    # But to make it more interesting (and like the blob in the image), let's weight edges
    # by how close their average scores are.
    
    # 1. Calculate Average Scores per Contestant
    contestant_stats = df.groupby("celebrity_name").agg({
        "judge_total": "mean",
        "fan_vote_share": "mean",
        "season": "first"
    }).reset_index()
    
    # Filter for top N contestants to keep graph readable but dense enough to resemble the example
    # N=50 is roughly enough to create a "blob"
    top_contestants = contestant_stats.head(100) 
    
    G = nx.Graph()
    
    print("Building network...")
    # Add nodes
    for idx, row in top_contestants.iterrows():
        G.add_node(row["celebrity_name"], 
                   season=row["season"], 
                   score=row["judge_total"])
    
    # Add edges based on "Score Similarity"
    # Connect every node to every other node if their score difference is small (threshold)
    names = top_contestants["celebrity_name"].values
    scores = top_contestants["judge_total"].values
    
    score_threshold = 0.5  # If avg scores are within 0.5 points, they are "linked" (similar performance tier)
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = abs(scores[i] - scores[j])
            if diff < score_threshold:
                # Weight is stronger if difference is smaller
                weight = 1.0 / (diff + 0.1)
                G.add_edge(names[i], names[j], weight=weight)
                
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Visualization
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)  # Force directed layout
    
    # Color nodes by Season or Score Cluster
    # Let's map seasons to colors
    seasons = [G.nodes[n]["season"] for n in G.nodes]
    
    # Draw huge blob background (simulate the colored cloud in the user image)
    # We can't do exact hull polygons easily without extra libs, but we can do large transparent nodes
    nx.draw_networkx_nodes(G, pos, 
                           node_size=800, 
                           node_color=seasons, 
                           cmap=plt.cm.Spectral, 
                           alpha=0.6)
    
    # Draw edges (thin, transparent)
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color="gray")
    
    # Labels (small)
    # nx.draw_networkx_labels(G, pos, font_size=6, alpha=0.8) 
    
    plt.title("Contestant Similarity Network (Score-based Clusters)", fontsize=16)
    plt.axis("off")
    
    out_path = FIG_DIR / "network_graph_demo.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Network graph saved to {out_path}")

if __name__ == "__main__":
    main()
