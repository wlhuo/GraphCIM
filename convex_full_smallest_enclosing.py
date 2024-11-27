# -*- encoding: utf-8 -*-
'''
@File    :   convex_full_smallest_enclosing.py
@Time    :   2024/11/27 15:03:06
@Author  :   wlhuo 
'''

import math
import numpy as np
import pandas as pd
from multiprocessing import Pool
from collections import defaultdict
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import minimize, differential_evolution, basinhopping, fmin



def read_and_group_cells(file_path):
    cell_dict = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            coords, cell_type = line.strip().split('\t')
            x, y = map(int, coords.split(':'))
            cell_type = float(cell_type)
            cell_dict[cell_type].append((x, y))

    return cell_dict

def smallest_enclosing_circle(points):
    def objective_function(center):
        radii = np.linalg.norm(points - center, axis=1)
        return np.max(radii)
    
    center_init = np.mean(points, axis=0)    
    center = fmin(objective_function, center_init, disp=False)
    radius = objective_function(center)
    
    return center, radius

def process_cell_dict(cell_dict):
    results = []
    for cell_id, points in cell_dict.items():
        hull = ConvexHull(points)
        cell_area = len(points)
        hull_points = [points[idx] for idx in hull.vertices]
        circle_center, circle_radius = smallest_enclosing_circle(hull_points)
        results.append({
                "cell_id": cell_id, 
                "centroid_x": circle_center[0], 
                "centroid_y": circle_center[1], 
                "radius": circle_radius,
                "cell_area": cell_area
            })
        
    
    return results


if __name__ == "__main__":

    input_path = './data/seq-scope/spot2cell.txt'
    count_path = "./data/seq-scope/counts.csv"
    cell_dict = read_and_group_cells(input_path)
    results = process_cell_dict(cell_dict)
    df = pd.DataFrame(results)
    counts_df = pd.read_csv(count_path)

    select_id = counts_df['Cell Type'].unique()
    df = df.sort_values('cell_id')
    filtered_df = df[df['cell_id'].isin(select_id)]
    output_df = pd.DataFrame({
        'X': filtered_df['centroid_x'],
        'Y': filtered_df['centroid_y']
    })

    output_df.to_csv("./data/seq-scope/coord.csv", index=False)