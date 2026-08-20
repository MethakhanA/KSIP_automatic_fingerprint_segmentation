import os
from glob import glob

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max

from blockbase_pipeline import BlockBaseFrameWork
from methlib.general import plot_all
from methlib.filter import FreqFilter

# Select 5 Top Peak
# vector -> Magnitude, Orientation (0-2pi), frequency (distance), harmonic (Peak at 2*frequency)
def orientation_estimation(block_img):
    # Drain water level
    qtile = np.quantile(block_img, [0.996, 0.997 ,0.998, 0.999])
    binned_img = np.digitize(block_img, bins=qtile)
    # thres_img = cv.inRange(block_img, np.array([np.]))
    return binned_img
    # pass

def draining_peak(block_img, radius_ban=3, max_peak_count=5, mean_radius_ban=(3, 16)):
    ''' - Finding 5 First peak by searching from maxima -> ban discovered peak -> find new peak
        - Add mean prevention thresholding.
    '''
    row, col = block_img.shape
    y_indices, x_indices = np.ogrid[:row, :col] # create coordinate grid
    
    peak_loc = []
    # plot_all(block_img, cmap='hot')
    x_center, y_center = col//2, row//2
    # ban_circular(block_img, x_center, y_center, mean_radius_ban, (y_indices, x_indices))
    block_img = ban_bandpass(block_img, mean_radius_ban[0], mean_radius_ban[1])
    for i in range(max_peak_count):
        peak_val = np.max(block_img)
        y_center, x_center = np.where(block_img==peak_val)
        peak_loc.append([y_center, x_center]) # Append peak location to list
        # Ban in radius
        for index in range(len(x_center)):
            ban_circular(block_img, x_center[index], y_center[index], radius_ban, (y_indices, x_indices))
        # plot_all(block_img, cmap='hot')
    return peak_loc
def local_multipeak(block_img, radius_ban=3, max_peak_count=5, mean_radius_ban=(3, 16)):
    block_img = ban_bandpass(block_img, mean_radius_ban=(3, 16))
    return peak_local_max(block_img, 3, num_peaks=10)

def ban_circular(block_img, centerx, centery, radius_ban, grid=None):
    row, col = block_img.shape
    if grid is None:
        y_indices, x_indices = np.ogrid[:row, :col]
    else:
        y_indices, x_indices = grid
    # Ban radius
    distance_sq = ((x_indices-centerx)**2 + (y_indices-centery)**2)
    mask = distance_sq <= radius_ban**2
    block_img[mask]=0
    return block_img
def ban_bandpass(block_img, radius1, radius2):
    filter = FreqFilter(block_img.shape)
    bandwidth = radius2-radius1
    bandcenter = radius1+(bandwidth/2)
    BPF = filter.getBPF(bandcenter, bandwidth)
    output = block_img*BPF
    return output
    
if __name__ == "__main__":
    path = r"C:\work\image_processing\latent_fingerprint\automatic_segment\data"
    for file in glob(os.path.join(path, '*')):
        img = cv.imread(file, 0)
        o_block_size = 64
        no_block_size = 32
        BBF = BlockBaseFrameWork(img, overlap_block_size=o_block_size, nonoverlap_block_size=no_block_size, zeromean=True, window_func='Gaussian')
        
        row_map_index_list, col_map_index_list = BBF.row_map_block_index_list, BBF.col_map_block_index_list
        
        BBF.stft()
        magnitude = BBF.getMagnitude()
        # for i in range(18):
        #     for j in range(18):
        i = 9
        j = 7
        block_img = magnitude[row_map_index_list[i]:row_map_index_list[i]+o_block_size, col_map_index_list[j]:col_map_index_list[j]+o_block_size]
        print(draining_peak(block_img))
        print(peak_local_max(block_img, 3, num_peaks=10))
        plot_all(block_img, cmap='hot')
        break
        
        # orient_field = BBF.apply_func_map(magnitude, orientation_estimation)
        # plt.imshow(orient_field, 'hot')
        # plt.show()