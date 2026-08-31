import numpy as np
from blockbase_pipeline import BlockBaseFrameWork
from  orientation_estimation import local_multipeak
from blockattribute import find_Attribute_multi
from utils.plot_all import plot_all
from utils.concave_hull import concave_hull
from grouping_framework import BlockGroup
from tqdm import tqdm


def map_clustering(vector_map, falloff_threshold=0.9, kernel=None, connectivity=8):
    "Iterative n-connectivity kernel block clustering"
    display_map = np.zeros_like(vector_map)
    if kernel is None:
        if connectivity==8:
            kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]], dtype=float)
        elif connectivity==4:
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=float)
    map_row, map_col = vector_map.shape
    k_row, k_col = kernel.shape
    if k_row%2==0 or k_col%2==0:
        raise ValueError("Kernel must have odd shape for it to have center!")
    ctr_k_row, ctr_k_col = k_row//2, k_col//2
    peak_pos = [(0, 0)]
    BG_list = [] # block group 1
    for pos in tqdm(peak_pos):
        r_idx, c_idx = pos
        p_val = vector_map[r_idx, c_idx] # Peak Value
        # Initiate row and column list
        BG = BlockGroup([r_idx], [c_idx])
        
        # iteratively find row and col index until not fit in falloff_threshold
        while True:
            for ctr_k_pos in BG.get_concave_hull():
                temp_kernel = kernel.copy()
                # center of kernel in map coordinate 
                ctr_row, ctr_col = ctr_k_pos
                start_row, stop_row =  ctr_row-ctr_k_row, ctr_row+ctr_k_row
                start_col, stop_col = ctr_col-ctr_k_col, ctr_col+ctr_k_col
                if start_row < 0:
                    rl_start_row = 0
                    temp_kernel = temp_kernel[(rl_start_row-start_row):, :]
                    print(temp_kernel)
                if stop_row >= map_row:
                    rl_stop_row = map_row-1
                    temp_kernel = temp_kernel[:(stop_row-rl_stop_row), :]
                    print(temp_kernel)
                if start_col < 0:
                    rl_start_col = 0
                if stop_col >= map_col:
                    rl_stop_col = map_col-1
                
                # print(f"\n{(stop_row-rl_stop_row)}\n")
                break
            break
        break
if __name__ == "__main__":
    import cv2 as cv
    from utils.freqfilter import FreqFilter
    filter = FreqFilter((32, 32))
    BPF = filter.getBPF(radius1=3, radius2=16)
    BPF = BPF.astype(np.float32)
    BPF = cv.GaussianBlur(BPF, (3, 3), sigmaX=0)
    # plot_all(BPF)
    map_clustering(BPF)