import matplotlib.pyplot as plt
import seaborn as sns

# Color Palette inspired by the reference images
COLORS = {
    "blue": "#5DADE2",       # Soft Blue (Figures 3, 10)
    "red": "#EC7063",        # Salmon Red (Figures 3, 10, 11)
    "teal": "#48C9B0",       # Teal/Green (Figure 15)
    "purple": "#AF7AC5",      # Purple (Figure 16)
    "orange": "#F5B041",     # Orange 
    "grey": "#95A5A6",       # Grey for neutral elements
    "dark_grey": "#34495E",  # Dark grey for text
    "grid": "#EAEDED"        # Very light grey for grid
}

# Categorical palette for comparison
PALETTE = [COLORS["blue"], COLORS["red"], COLORS["teal"], COLORS["purple"], COLORS["orange"]]

def set_style():
    """Calculates and sets the matplotlib style parameters for publication-quality figures."""
    
    # Use seaborn as a base for better defaults
    sns.set_theme(style="white", palette=PALETTE)
    
    # Custom adjustments
    plt.rcParams.update({
        # Font settings (High-end Academic)
        "font.family": "sans-serif",
        # Prioritize Chinese fonts for Windows to ensure glyphs are found
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "sans-serif"], 
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,

        # Text color
        "text.color": COLORS["dark_grey"],
        "axes.labelcolor": COLORS["dark_grey"],
        "xtick.color": COLORS["dark_grey"],
        "ytick.color": COLORS["dark_grey"],

        # Axes and Grid
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Figure formatting
        "figure.figsize": (10, 6),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        
        # Lines and Markers
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        
        # Bar charts
        "patch.linewidth": 0, # No border on bars by default
        "patch.edgecolor": "none"
    })

def save_fig(fig, path):
    """Save figure with consistent settings."""
    # Ensure tight layout
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
