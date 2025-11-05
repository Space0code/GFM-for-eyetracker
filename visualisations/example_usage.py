#!/usr/bin/env python3
"""
Example usage of the visualization script.
Run this from the project root directory.
"""

# Example commands to run the visualization:

# 1. Visualize predictions for a single CSV file (display plot)
# python visualisations/visualize_predictions.py data/processed/cog-load/s_001.csv

# 2. Visualize and save the plot
# python visualisations/visualize_predictions.py data/processed/cog-load/s_001.csv --save visualisations/s_001_predictions.png

# 3. Use a different model checkpoint
# python visualisations/visualize_predictions.py data/processed/cog-load/s_001.csv --model path/to/other/model.pt

print("Run the visualization with:")
print("python visualisations/visualize_predictions.py data/processed/cog-load/s_001.csv")