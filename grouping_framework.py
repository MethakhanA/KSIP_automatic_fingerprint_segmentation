import numpy as np
import cv2 as cv
from tqdm import tqdm

from blockbase_pipeline import BlockBaseFrameWork
from  orientation_estimation import local_multipeak
from blockattribute import find_Attribute_multi

from utils.concave_hull import concave_hull
class BlockGroup:
    def __init__(self, row_index_list, col_index_list, block_attribute_list=None):
        self.rw_idx_lst = row_index_list
        self.cl_idx_lst = col_index_list
        self.blk_attrib_lst = block_attribute_list
        self.__gen_id()
    def __gen_id(self):
        group_id = {}
        blk_attrib_lst = self.blk_attrib_lst
        attrib_idx = 0
        for r_idx in self.rw_idx_lst:
            for c_idx in self.cl_idx_lst:
                if blk_attrib_lst is None:
                    group_id[(r_idx, c_idx)] = None
                else:
                    group_id[(r_idx, c_idx)] = blk_attrib_lst[attrib_idx]
                attrib_idx += 1
        self.group_id = group_id
    def add_member(self, row_index, col_index, attribute=None):
        self.group_id[(row_index, col_index)] = attribute
    def remove_member(self, row_index, col_index):
        del self.group_id[(row_index, col_index)]
    def generate_activation_map(self, max_row, max_col):
        atv_map = np.zeros((max_row, max_col), dtype=bool)
        for pos in self.group_id:
            row_idx, col_idx = pos
            atv_map[row_idx, col_idx] = True
        return atv_map
    def get_member_attribute(self, row_index, col_index):
        return self.group_id[(row_index, col_index)]
    def get_member_list(self):
        return [id for id in self.group_id]
    def get_concave_hull(self):
        points = self.get_member_list()
        hull_points = concave_hull(points)
        if hull_points is None:
            return points
        return hull_points
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
    peak_pos = local_multipeak(vector_map, radius_ban=3, max_peak_count=1)
    BG_list = [] # block group 1
    for pos in tqdm(peak_pos):
        r_idx, c_idx = pos
        p_val = vector_map[r_idx, c_idx]# Peak Value
        # Initiate row and column list
        BG = BlockGroup([r_idx], [c_idx])
        
        # iteratively find row and col index until not fit in falloff_threshold
        index = 0
        delta = BG.get_member_list()
        while True:
            change = False
            for ctr_pos in delta:
                temp_kernel = kernel.copy()
                # center of kernel in map coordinate
                ctr_row, ctr_col = ctr_pos
                display_map[ctr_row, ctr_col] = 1
                start_row, stop_row =  ctr_row-ctr_k_row, ctr_row+ctr_k_row
                start_col, stop_col = ctr_col-ctr_k_col, ctr_col+ctr_k_col
                if start_row < 0:
                    rl_start_row = 0
                    temp_kernel = temp_kernel[(rl_start_row-start_row):, :]
                else:
                    rl_start_row = start_row
                if stop_row >= map_row:
                    rl_stop_row = map_row-1
                    temp_kernel = temp_kernel[:-(stop_row-rl_stop_row), :]
                else:
                    rl_stop_row = stop_row
                if start_col < 0:
                    rl_start_col = 0
                    temp_kernel = temp_kernel[:, (rl_start_col-start_col):]
                else:
                    rl_start_col = start_col
                if stop_col >= map_col:
                    rl_stop_col = map_col-1
                    temp_kernel = temp_kernel[:, :-(stop_col-rl_stop_col)]
                else:
                    rl_stop_col = stop_col
                
                output = temp_kernel*vector_map[rl_start_row:rl_stop_row+1, rl_start_col:rl_stop_col+1]
                
                thres = falloff_threshold*vector_map[ctr_row, ctr_col]
                    
                _, output = cv.threshold(output, thres, 1, cv.THRESH_BINARY)

                rel_ctr_row = ctr_row - rl_start_row
                rel_ctr_col = ctr_col - rl_start_col
                output[rel_ctr_row, rel_ctr_col] = 0
                rel_pos = np.where(output>0)
                if len(rel_pos[0])==0:
                    # Do nothing, skip to next loop
                    continue
                for i in range(len(rel_pos[0])):
                    rl_row, rl_col = rl_start_row+rel_pos[0][i], rl_start_col+rel_pos[1][i]
                    if not (rl_row, rl_col) in BG.get_member_list():
                        BG.add_member(rl_row, rl_col)
                        if not change:
                            change = True
            temp = BG.get_member_list()
            set_delta = set(delta)
            delta = [item for item in temp if item not in set_delta]
            if not change:
                # There is no change
                break
        BG_list.append(BG)
    # plot_all([vector_map, display_map])
    return BG_list, display_map

# if __name__ == "__main__":
#     # Need Testing and debug
#     # path = r"D:\work\image_processing\Latent_fingerprint\segment\data"
#     pass