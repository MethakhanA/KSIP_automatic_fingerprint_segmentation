import numpy as np
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
    k_row, k_col = k_row-1, k_col-1
    ctr_row, ctr_col = k_row//2, k_col//2
    peak_pos = local_multipeak(vector_map, radius_ban=3, max_peak_count=10)
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
                ctr_k_row, ctr_k_col = ctr_k_pos
                display_map[ctr_k_row, ctr_k_col] = 1
                start_row = ctr_k_row-ctr_row
                stop_row = start_row+k_row
                start_col = ctr_k_col-ctr_col
                stop_col= start_col+k_col
                # If the kernel is out of bound
                if start_row<0:
                    start_row = 0
                    temp_kernel = temp_kernel[k_row-(stop_row-start_row):, :]
                if start_col<0:
                    start_col = 0
                    temp_kernel = temp_kernel[:, k_col-(stop_col-start_col):]
                if stop_row>=map_row:
                    stop_row = map_row-1
                    temp_kernel = temp_kernel[:stop_row-start_row, :]
                if stop_col>=map_col:
                    stop_col = map_col-1
                    temp_kernel = temp_kernel[:, :stop_col-start_col]
                # Now time it with the real
                output = temp_kernel*vector_map[start_row:stop_row+1, start_col:stop_col+1]
                print(output)
                # Check condition
                output[output<(falloff_threshold*vector_map[ctr_k_row, ctr_k_col])]=0
                output[output>0]=1
                output = output
                r_map_k_lst = range(start_row, stop_row)
                c_map_k_lst = range(start_col, stop_col)
                for i in range(len(r_map_k_lst)):
                    for j in range(len(c_map_k_lst)):
                        if output[i, j]==1:
                            if not ((r_map_k_lst[i], c_map_k_lst[j]) in BG.get_member_list()):
                                # Then Add this to the Block group
                                BG.add_member(r_map_k_lst[i], c_map_k_lst[j])
                                
                # Add condition for code to exit. (No more fall off)
            print(np.sum(output))
            plot_all(display_map)
            # break
            if np.sum(output)<=1:
                break
        BG_list.append(BG)
    return BG_list

if __name__ == "__main__":
    # Need Testing and debug
    # path = r"D:\work\image_processing\Latent_fingerprint\segment\data"
    import os
    import cv2 as cv
    from tqdm import tqdm
    from glob import glob
    from crossing_field import ban_bandpass_gaussian
    from utils.plot_all import plot_all
    from utils.kurtosis import fft_kurtosis
    out_path = r"D:\work\image_processing\Latent_fingerprint\segment\data_TV"
    for file in tqdm(glob(os.path.join(out_path, '*'))):
        img = cv.imread(file, 0) # This is a Texture TV image.
        o_block_size = 64
        no_block_size = 16
        bp_r = (3, 16)
        gss_f_size = 3
        
        BBF = BlockBaseFrameWork(img, overlap_block_size=o_block_size, nonoverlap_block_size=no_block_size, zeromean=True, window_func='Gaussian', blur_edge=True)
        row_map_index, col_map_index = BBF.row_map_block_index_list, BBF.col_map_block_index_list
        pad_img = BBF.get_img()
        BBF.stft()
        magnitude = BBF.getMagnitude().astype(np.float32)
        # Apply Gaussian Bandpass
        magnitude = BBF.apply_func_map(magnitude, ban_bandpass_gaussian, bp_r[0], bp_r[1], gss_f_size)
        # create kurtosis map
        ks_map = np.zeros((len(row_map_index), len(col_map_index)))
        ks_map = BBF.apply_func_map(magnitude, fft_kurtosis, output_is_img=False, output_vector=ks_map)
        # plot_all([pad_img, ks_map], cmap=['gray', 'hot'])
        # # create Orientation map
        # orientation_map = np.zeros((len(row_map_index), len(col_map_index), 4))
        # orientation_map = BBF.apply_func_map(magnitude, find_Attribute_multi, output_is_img=False, output_vector=orientation_map)
        # plot_all([orientation_map[:, :, i] for i in range(4)])
        group_list = map_clustering(ks_map, falloff_threshold=0.1)
        print(group_list)