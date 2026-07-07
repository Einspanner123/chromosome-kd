#!/usr/bin/env python3
"""Quick runner for Fig 12 only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Need the project root for data paths (datasets use relative data/ paths)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

from dataset_comparison import build_dataframes, plot_category_per_image

dfs = build_dataframes()
plot_category_per_image(dfs)
print("Fig 12 done")
