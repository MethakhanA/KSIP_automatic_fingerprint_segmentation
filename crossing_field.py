import os
from glob import glob

import numpy as np
import cv2 as cv
from tqdm import tqdm

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

if __name__ == "__main__":
    out_path = r"D:\work\image_processing\Latent_fingerprint\segment\data"
    # out_path = r"D:\work\image_processing\Latent_fingerprint\segment\data_TV"
    # out_path = r"C:\work\image_processing\latent_fingerprint\automatic_segment\data"
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
        index = 15
        # plot_all(magnitude[row_map_index[index]:row_map_index[index]+o_block_size, col_map_index[index]:col_map_index[index]+o_block_size], cmap='hot', title_list=["Original Magnitude"])
        # Apply Gaussian Bandpass
        magnitude = BBF.apply_func_map(magnitude, ban_bandpass_gaussian, bp_r[0], bp_r[1], gss_f_size)
        # plot_all(magnitude[row_map_index[index]:row_map_index[index]+o_block_size, col_map_index[index]:col_map_index[index]+o_block_size], cmap='hot', title_list=["Gaussian Bandpass Magnitude"])
        # create kurtosis map
        ks_map = np.zeros((len(row_map_index), len(col_map_index)))
        ks_map = BBF.apply_func_map(magnitude, fft_kurtosis, output_is_img=False, output_vector=ks_map)
        block_pad = np.array(BBF.fp_pad)//no_block_size
        ks_map = ks_map[block_pad[0]:-block_pad[1], block_pad[2]:-block_pad[3]]
        pad_img = pad_img[BBF.fp_pad[0]:-BBF.fp_pad[1], BBF.fp_pad[2]:-BBF.fp_pad[3]]
        # plot_all([pad_img, ks_map], cmap=['gray', 'hot'])
        map_clustering_watershed(ks_map)
        # plot_all([pad_img, ks_map], cmap=['gray', 'hot']) # Show kurtosis map
        # cluster_list, cluster_map_list = map_clustering(ks_map, 0.9)
        # cluster_map_list.insert(0, ks_map)

        # plot_all([item for item in cluster_map_list], cmap='hot')
        # # break
        
        # create Orientation map
        # orientation_map = np.empty((len(row_map_index), len(col_map_index)), dtype=object)
        # orientation_map = BBF.apply_func_map(magnitude, find_Attribute_multi, 10, False, output_is_img=False, output_vector=orientation_map)
        
        # plot_all([orientation_map[:, :, i] for i in range(2, 6)])
        # print(orientation_map)
        # break