import numpy as np
from collections import deque
import cv2 as cv
from tqdm import tqdm

# from skimage.feature import peak_local_max

from blockbase_pipeline import BlockBaseFrameWork
from  orientation_estimation import local_multipeak, banning_peak
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
def map_clustering_n_iterative(vector_map, falloff_threshold=0.9, kernel=None, connectivity=8, custom_peak_pos=None):
    "Iterative n-connectivity kernel block clustering"
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
    if custom_peak_pos is None:
        peak_pos = local_multipeak(vector_map, radius_ban=2, max_peak_count=10)
    else:
        peak_pos = custom_peak_pos # put your custom peak pos here
    if peak_pos is None:
        return None, None
    BG_list = [] # block group 
    display_map_list = []
    for pos in tqdm(peak_pos):
        display_map = np.zeros_like(vector_map)
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
            # plot_all([vector_map, display_map]) # Uncomment ts
            if not change:
                # There is no change
                break
        BG_list.append(BG)
        display_map_list.append(display_map)
    return BG_list, display_map_list

def map_clustering_watershed(image, connectivity=8, merge_threshold=0.5):
    """Clusters pixels hierarchically from peaks downward (monotonically non-increasing),

    excluding 0-intensity background, with per-step visualization.

    :param image: 2D numpy array (grayscale image)
    :param connectivity: 4 or 8 for neighbor checking
    :param merge_threshold: Fraction of overlap required to merge two
    clusters
    :return: List of clusters, each containing a list of (y, x) tuples
    """
    rows, cols = image.shape

    if connectivity == 4:
        # 4 connectivity
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        # 8 Connectivity
        offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    def get_neighbors(y, x):
        neighbors = []
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                neighbors.append((ny, nx))
        return neighbors

    # 1. Filter out zero-intensity background and sort coordinates descending by intensity
    y_coords, x_coords = np.nonzero(image)
    intensities = image[y_coords, x_coords]

    sort_idx = np.argsort(intensities)[::-1]
    sorted_pixels = list(zip(y_coords[sort_idx], x_coords[sort_idx]))

    visited_by_any = set()
    clusters = []

    # 2 & 3. Identify Peaks and Perform Downward Region Growing
    for peak_y, peak_x in sorted_pixels:
        if (peak_y, peak_x) in visited_by_any:
            continue

        current_cluster_set = set([(peak_y, peak_x)])
        queue = deque([(peak_y, peak_x)])

        while queue:
            curr_y, curr_x = queue.popleft()
            curr_intensity = image[curr_y, curr_x]

            # # --- VISUALIZATION HOOK ---
            # vis_array = np.zeros_like(image)
            # for cy, cx in current_cluster_set:
            #     vis_array[cy, cx] = image[cy, cx]

            # plot_all(vis_array)
            # # --------------------------

            for ny, nx in get_neighbors(curr_y, curr_x):
                n_intensity = image[ny, nx]

                # Exclude background pixels and already included pixels
                if n_intensity == 0 or (ny, nx) in current_cluster_set:
                    continue

                # Monotonic rule: Neighbor intensity must be less than or equal to current pixel
                if n_intensity <= curr_intensity:
                    current_cluster_set.add((ny, nx))
                    visited_by_any.add((ny, nx))
                    queue.append((ny, nx))

        clusters.append(current_cluster_set)

    # 4. Merge Similar Clusters
    merged_clusters = []
    for cluster in clusters:
        merged = False
        for i, existing_cluster in enumerate(merged_clusters):
            intersection = cluster.intersection(existing_cluster)

            min_size = min(len(cluster), len(existing_cluster))
            if len(intersection) / min_size > merge_threshold:
                merged_clusters[i] = existing_cluster.union(cluster)
                merged = True
                break

        if not merged:
            merged_clusters.append(cluster)
    # return [list(c) for c in merged_clusters]
    # return [BlockGroup([item[0] for item in list(c)], [item[1] for item in list(c)]) for c in merged_clusters]
    return [BlockGroup(*zip(*c)) for c in merged_clusters]
# Example
if __name__ == "__main__":
    from utils.freqfilter import FreqFilter
    from utils.plot_all import plot_all
    # Create example
    size = 32
    radius1, radius2 = 3, 16
    filter = FreqFilter((size, size))
    BPF = filter.getBPF(radius1=radius1, radius2=radius2)
    BPF = BPF.astype(np.float32)
    BPF = cv.GaussianBlur(BPF, (3, 3), sigmaX=0)
    # plot_all(BPF)
    dat_list = map_clustering_watershed(BPF)

    # print(dat_list[0].get_member_list())