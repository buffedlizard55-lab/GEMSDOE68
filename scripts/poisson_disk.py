#!/usr/bin/env python3
"""
Fast Poisson-disk thinning on a 2D grid.
Uses a grid-based acceleration for O(n) expected time.
"""
import numpy as np
from scipy.spatial import cKDTree
import time

def poisson_disk_thin(coords, probs, min_dist, max_points=None):
    """
    Greedy Poisson-disk thinning: sort by probability descending, 
    keep point if far enough from all previously kept points.
    
    Uses a cell grid for fast neighbor lookup.
    
    Parameters:
        coords: (N, 2) array of (y, x) coordinates
        probs: (N,) array of probabilities (higher = keep first)
        min_dist: minimum distance between kept points (in pixels)
        max_points: stop after this many points (None = no limit)
    
    Returns:
        kept_indices: indices into coords of kept points
    """
    N = len(coords)
    if N == 0:
        return np.array([], dtype=int)
    
    # Sort by probability descending
    order = np.argsort(-probs)
    coords_sorted = coords[order]
    
    # Cell grid for fast neighbor checking
    cell_size = min_dist  # each cell is min_dist x min_dist
    H, W = 3730, 3292  # grid dimensions
    grid_h = int(np.ceil(H / cell_size)) + 1
    grid_w = int(np.ceil(W / cell_size)) + 1
    
    # occupied[y, x] = True if cell has a kept point
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    
    # Search radius in cells
    search_r = int(np.ceil(min_dist / cell_size)) + 1
    
    kept = []
    kept_coords_list = []
    
    for i in range(N):
        y, x = coords_sorted[i]
        
        # Cell of this point
        cy = int(y / cell_size)
        cx = int(x / cell_size)
        
        # Check if any neighbor cell is occupied
        too_close = False
        y_min = max(0, cy - search_r)
        y_max = min(grid_h, cy + search_r + 1)
        x_min = max(0, cx - search_r)
        x_max = min(grid_w, cx + search_r + 1)
        
        if occupied[y_min:y_max, x_min:x_max].any():
            # Need to check exact distance
            if len(kept_coords_list) > 0:
                arr = np.array(kept_coords_list)
                tree = cKDTree(arr)
                d, _ = tree.query([y, x])
                if d < min_dist:
                    too_close = True
        
        if not too_close:
            kept.append(i)
            kept_coords_list.append([y, x])
            occupied[cy, cx] = True
            
            if max_points is not None and len(kept) >= max_points:
                break
    
    # Map back to original indices
    kept_original = order[kept]
    return kept_original
