import numpy as np
from blockbase_pipeline import BlockBaseFrameWork
from  orientation_estimation import local_multipeak
from blockattribute import find_Attribute_multi

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
    
def map_clustering(vector_map, falloff_threshold=0.9, kernel=None, connectivity=8):
    "Iterative n-connectivity kernel block clustering"
    
    if kernel is None:
        if connectivity==8:
            kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]], dtype=bool)
            kernel = np.all()
        elif connectivity==4:
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=bool)
    row, col = kernel.shape
    c_row, c_col = row//2+1, col//2+1
    if row%2==0 or col%2==0:
        raise ValueError("Kernel must have odd shape for it to have center!")
    peak_pos = local_multipeak(vector_map, radius_ban=3, max_peak_count=10)
    for pos in peak_pos:
        r_idx, c_idx = pos
        p_val = vector_map[r_idx, c_idx] # Peak Value
        # Initiate row and column list
        r_idx_lst = [r_idx]
        c_idx_lst = [c_idx]
        # iteratively find row and col index until not fit in falloff_threshold
        while True:
            
            # Put kernel into position
            for r_k_idx in range(row):
                for c_k_idx in range(col):
                    if kernel[r_k_idx][c_k_idx]:
                        
            pass