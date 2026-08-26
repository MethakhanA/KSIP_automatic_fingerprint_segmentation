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
from blockattribute import BlockAttribute
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
    
def find_Attribute_multi(block_img):
    peak_pos = local_multipeak(block_img, 3, 1)
    # peak_pos = banning_peak(block_img, 3, 1)
    if peak_pos is False:
        return 0.0 ,0.0, 0.0, 0.0
    output_vector = np.zeros((len(peak_pos), 4))
    for index in range(len(peak_pos)):
        BA = BlockAttribute(block_img, peak_pos[index])
        direction = BA.find_direction()
        magnitude = BA.find_magnitude()
        frequency = BA.find_distance()
        harmonic = BA.find_harmonic()
        output_vector[index] = np.array([direction, magnitude, frequency, harmonic])
    return output_vector



if __name__ == "__main__":
    # path = r"D:\work\image_processing\Latent_fingerprint\segment\data"
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
        # create Orientation map
        orientation_map = np.zeros((len(row_map_index), len(col_map_index), 4))
        orientation_map = BBF.apply_func_map(magnitude, find_Attribute_multi, output_is_img=False, output_vector=orientation_map)
        plot_all([orientation_map[:, :, i] for i in range(4)])