import os
from glob import glob

import numpy as np
import cv2 as cv


from blockbase_pipeline import BlockBaseFrameWork

from utils.plot_all import plot_all
'''
Crossing point field framework
- from BBF generate STFT map
- generate 
    1. field containing orientation and its property
    2. Vector recording crossing point and its orientation


'''


if __name__ == "__main__":
    path = r"D:\work\image_processing\Latent_fingerprint\segment\data"
    for file in glob(os.path.join(path, '*')):
        img = cv.imread(file, 0)
        o_block_size = 64
        no_block_size = 32
        BBF = BlockBaseFrameWork(img, overlap_block_size=o_block_size, nonoverlap_block_size=no_block_size, zeromean=True, window_func='Gaussian', blur_edge=True)