import os
from glob import glob

import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.TV import TV_preprocessing
from utils.plot_all import plot_all
from utils.freqfilter import FreqFilter
from utils.kurtosis import fft_kurtosis

from blockbase_pipeline import BlockBaseFrameWork
from orientation_estimation import banning_peak, local_multipeak
from blockattribute import find_Attribute_multi
from grouping_framework import map_clustering_watershed
'''
Crossing point field framework
- from BBF generate STFT map
- generate 
    1. field containing orientation and its property
    2. Vector recording crossing point and its orientation


'''


def ban_bandpass_gaussian(block_img, radius1, radius2, filtersize=3):
    filter = FreqFilter(block_img.shape)
    BPF = filter.getBPF(radius1=radius1, radius2=radius2)
    BPF = BPF.astype(np.float32)
    BPF = cv.GaussianBlur(BPF, (filtersize, filtersize), sigmaX=0)
    output = block_img*BPF
    return output

# write a crossing field framework
# from watershed_clustering_test import cluster_pixels
def find_orientation_crossings(ori_array, d=5.0, dist_tol=1.5, cluster_tol=3.0, activation_map=None):
    H = len(ori_array)
    W = len(ori_array[0]) if H > 0 else 0
    
    valid_cells = []
    thetas = []
    
    # 1. Extract valid cells considering both ori_array and activation_map
    for y in range(H):
        for x in range(W):
            if ori_array[y][x] is not None:
                if activation_map is None or activation_map[y][x]:
                    valid_cells.append((y, x))
                    thetas.append(ori_array[y][x])
                
    N = len(valid_cells)
    if N < 3:
        return []

    valid_cells = np.array(valid_cells)
    ys = valid_cells[:, 0]
    xs = valid_cells[:, 1]
    
    thetas = np.array(thetas, dtype=float)
    A = np.sin(thetas)
    B = -np.cos(thetas)
    C = xs * A + ys * B
    
    raw_intersections = []

    # 2. Vectorized pairwise intersections safely handling determinants
    for i in range(N - 1):
        A_i, B_i, C_i = A[i], B[i], C[i]
        y_i, x_i = ys[i], xs[i]
        
        A_j, B_j, C_j = A[i+1:], B[i+1:], C[i+1:]
        ys_j, xs_j = ys[i+1:], xs[i+1:]
        
        Det = A_i * B_j - A_j * B_i
        valid_mask = np.abs(Det) > 1e-6
        
        if not np.any(valid_mask):
            continue
            
        Det_v = Det[valid_mask]
        B_j_v, C_j_v = B_j[valid_mask], C_j[valid_mask]
        A_j_v = A_j[valid_mask]
        xs_j_v, ys_j_v = xs_j[valid_mask], ys_j[valid_mask]
            
        cross_x = (C_i * B_j_v - C_j_v * B_i) / Det_v
        cross_y = (A_i * C_j_v - A_j_v * C_i) / Det_v
        
        in_bounds = (cross_x >= -W) & (cross_x <= 2*W) & \
                    (cross_y >= -H) & (cross_y <= 2*H)
                    
        dist_from_i = np.hypot(cross_x - x_i, cross_y - y_i)
        dist_from_j = np.hypot(cross_x - xs_j_v, cross_y - ys_j_v)
        valid_dist = (dist_from_i >= d) & (dist_from_j >= d)
        
        final_mask = in_bounds & valid_dist
        
        if np.any(final_mask):
            valid_xs = cross_x[final_mask]
            valid_ys = cross_y[final_mask]
            for vx, vy in zip(valid_xs, valid_ys):
                raw_intersections.append((vy, vx))
                
    if not raw_intersections:
        return []

    # 3. Cluster intersections
    clusters = []
    cluster_centers = []
    
    for pt in raw_intersections:
        pt_arr = np.array(pt)
        if not clusters:
            clusters.append([pt_arr])
            cluster_centers.append(pt_arr)
            continue
            
        centers_arr = np.array(cluster_centers)
        dists = np.linalg.norm(centers_arr - pt_arr, axis=1)
        min_idx = np.argmin(dists)
        
        if dists[min_idx] <= cluster_tol:
            clusters[min_idx].append(pt_arr)
            cluster_centers[min_idx] = np.mean(clusters[min_idx], axis=0)
        else:
            clusters.append([pt_arr])
            cluster_centers.append(pt_arr)

    # 4. Find valid clusters with >= 3 contributors
    candidate_output = []
    for center in cluster_centers:
        cy, cx = center
        perp_dists = np.abs(A * cx + B * cy - C)
        eucl_dists = np.hypot(cx - xs, cy - ys)
        
        contributors_mask = (perp_dists <= dist_tol) & (eucl_dists >= d)
        
        if np.sum(contributors_mask) >= 3:
            contrib_indices = np.where(contributors_mask)[0]
            crossing_tuple = (float(cy), float(cx))
            blocks = tuple(sorted([(int(ys[idx]), int(xs[idx])) for idx in contrib_indices]))
            candidate_output.append((crossing_tuple, blocks))

    # 5. Cleanup
    unique_results = {}
    for cross_pt, blocks in candidate_output:
        if blocks not in unique_results:
            unique_results[blocks] = [cross_pt]
        else:
            unique_results[blocks].append(cross_pt)
            
    final_output = []
    for blocks, points in unique_results.items():
        avg_y = sum(p[0] for p in points) / len(points)
        avg_x = sum(p[1] for p in points) / len(points)
        final_output.append( ((avg_y, avg_x),) + blocks )
            
    return final_output


def visualize_crossings(ori_array, crossings, extension_length=8.0, background_image=None, cmap='gray'):
    H = len(ori_array)
    W = len(ori_array[0]) if H > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-5, W + 5)
    ax.set_ylim(-5, H + 5)
    
    # Render background image if provided, perfectly aligned with the grid
    if background_image is not None:
        ax.imshow(background_image, cmap=cmap, origin='upper', 
                  extent=[-0.5, W-0.5, H-0.5, -0.5], alpha=0.75)
    else:
        ax.invert_yaxis()
    
    # 1. Setup explicit pixel grid
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.tick_params(which='minor', length=0)
    
    # 2. Plot Orientation Field as headless lines
    for y in range(H):
        for x in range(W):
            theta = ori_array[y][x]
            if theta is not None:
                dx = np.cos(theta) * 0.45
                dy = np.sin(theta) * 0.45
                ax.plot([x - dx, x + dx], [y - dy, y + dy], color='white' if background_image is not None else 'dimgray', linewidth=1.5)
    
    # 3. Plot Crossings and Extended Lines
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']
    for i, crossing_data in enumerate(crossings):
        cy, cx = crossing_data[0]
        blocks = crossing_data[1:]
        color = colors[i % len(colors)]
        
        ax.scatter(cx, cy, color=color, marker='*', s=300, edgecolor='black', zorder=5, 
                   label=f'Cross {i+1} ({cx:.1f}, {cy:.1f})')
        
        for (by, bx) in blocks:
            ax.scatter(bx, by, color=color, marker='s', s=40, edgecolor='black', zorder=4)
            
            # Extend line past crossing point
            dist = np.hypot(cx - bx, cy - by)
            if dist > 0:
                ex = cx + ((cx - bx) / dist) * extension_length
                ey = cy + ((cy - by) / dist) * extension_length
                ax.plot([bx, ex], [by, ey], color=color, linestyle='-', alpha=0.8, linewidth=2.0)

    ax.set_title("Orientation Field Crossings with Background Image")
    ax.set_xlabel("X (Image Width)")
    ax.set_ylabel("Y (Image Height)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()
def exclude_boundary(img):
    img[0, :] = False
    img[-1, :] = False
    img[:, 0] = False  
    img[:, -1] = False
    return img

if __name__ == "__main__":
    # from crossingfield_test import find_orientation_crossings, visualize_crossings
    from grouping_framework import BlockGroup
    out_path = r"D:\work\image_processing\Latent_fingerprint\segment\data"
    # out_path = r"D:\work\image_processing\Latent_fingerprint\segment\data_TV"
    # out_path = r"C:\work\image_processing\latent_fingerprint\automatic_segment\data"
    # ---- Define Params ----
    o_block_size = 64
    no_block_size = 16
    bp_r = (3, 16)
    gss_f_size = 3
    # ---- ---- ----
    for file in tqdm(glob(os.path.join(out_path, '*'))):
        img = cv.imread(file, 0) # This is a Texture TV image.
        # ---- Run BlockBase PipeLine ----
        BBF = BlockBaseFrameWork(img, overlap_block_size=o_block_size, nonoverlap_block_size=no_block_size, zeromean=True, window_func='Gaussian', blur_edge=True)
        row_map_index, col_map_index = BBF.row_map_block_index_list, BBF.col_map_block_index_list
        pad_img = BBF.get_img()
        BBF.stft()
        magnitude = BBF.getMagnitude().astype(np.float32)
        # plot_all(magnitude[row_map_index[index]:row_map_index[index]+o_block_size, col_map_index[index]:col_map_index[index]+o_block_size], cmap='hot', title_list=["Original Magnitude"])
        # ---- ---- ----
        
        # ---- Apply Gaussian Bandpass ----
        magnitude = BBF.apply_func_map(magnitude, ban_bandpass_gaussian, bp_r[0], bp_r[1], gss_f_size)
        # plot_all(magnitude[row_map_index[index]:row_map_index[index]+o_block_size, col_map_index[index]:col_map_index[index]+o_block_size], cmap='hot', title_list=["Gaussian Bandpass Magnitude"])
        
        # ---- Create kurtosis map ----
        ks_map = np.zeros((len(row_map_index), len(col_map_index)))
        ks_map = BBF.apply_func_map(magnitude, fft_kurtosis, output_is_img=False, output_vector=ks_map)
        # ---- ---- ----
        
        # ---- Crop Kurtosis and picture for better Visualization ----
        block_pad = np.array(BBF.fp_pad)//no_block_size
        ks_map = ks_map[block_pad[0]:-block_pad[1], block_pad[2]:-block_pad[3]]
        pad_img = pad_img[BBF.fp_pad[0]:-BBF.fp_pad[1], BBF.fp_pad[2]:-BBF.fp_pad[3]]
        # plot_all([pad_img, ks_map], cmap=['gray', 'hot'])

        # # /// C1: ---- Create Activation map from watershed clustering
        # bg_list = map_clustering_watershed(ks_map)
        # max_pos_list = [item.max_pos for item in bg_list] # each item is a Block Group
        # mp_grp = BlockGroup([itm1[0] for itm1 in max_pos_list], [itm2[1] for itm2 in max_pos_list])
        # # ---- ---- ----
        
        # /// C2: ---- Create Activation map from localmultipeak and banningpeak
        max_pos_list = local_multipeak(ks_map, mean_radius_ban=None)
        # max_pos_list = banning_peak(ks_map, mean_radius_ban=None)
        row_mp_list, col_mp_list = [item[0] for item in max_pos_list], [item[1] for item in max_pos_list]
        mp_grp = BlockGroup(row_mp_list, col_mp_list)
        activation_map = mp_grp.generate_activation_map(ks_map.shape[0], ks_map.shape[1])
        # ---- ---- ----
        
        # ---- Create Activation Map
        activation_map = mp_grp.generate_activation_map(ks_map.shape[0], ks_map.shape[1])
        activation_map = exclude_boundary(activation_map)        
        plot_all([pad_img, ks_map, activation_map], cmap=['gray', 'hot', 'gray'])

        # ---- Create Attribute map ----
        orientation_map = np.empty((len(row_map_index), len(col_map_index)), dtype=np.ndarray)
        orientation_map = BBF.apply_func_map(magnitude, find_Attribute_multi, 10, False, output_is_img=False, output_vector=orientation_map)
        # ---- ---- ----
        
        # ---- Get orientation map from Attribute map ----
        o_map = np.zeros((len(row_map_index), len(col_map_index)))
        for i in range(len(orientation_map)):
            for j in range(len(orientation_map[i])):
                if orientation_map[i, j] is None:
                    o_map[i, j] = None
                else:
                    # print(orientation_map[i, j][0][0])
                    o_map[i, j] = orientation_map[i, j][0][0]
        o_map = o_map[block_pad[0]:-block_pad[1], block_pad[2]:-block_pad[3]]
        # print(o_map.shape, activation_map.shape)
        # ---- ---- ----

        # ---- Find Crossing Field ----
        result = find_orientation_crossings(o_map, activation_map=activation_map)
        visualize_crossings(o_map, result, background_image=pad_img, cmap='gray')
        # ---- ---- ----