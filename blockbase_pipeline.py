# Path related Lib
import os
import math
from glob import glob

# Operation related Lib
import numpy as np
from tqdm import tqdm

# Image related Lib
import cv2 as cv
import matplotlib.pyplot as plt

# Local Lib
# from methlib.general import plot_all, normalize_range
# from methlib.intensity_transform import log_transform
from utils.fourier import Fourier2D
from utils.blur_edge import blurEdge

class BlockBaseFrameWork:
    def __init__(self, img=None, overlap_block_size:int=64, nonoverlap_block_size:int=32, zeromean:bool=False, window_func=None, blur_edge=False, blur_size=21, erode_size=21):
        """Initialize the block-based framework for image processing."""
        self.__img = img
        self.__zeromean = zeromean
        self.__window_func = window_func
        self.__blur_edge = blur_edge
        self.__blur_edge_size = (blur_size, erode_size)
        
        # Initialize map holders
        self.__spa_map_img = None
        self.__freq_map_img = None
        self.__phase_map_img = None
        self.__mean_arr = None
        self.__output_img = None

        self.__padding(overlap_block_size, nonoverlap_block_size) # Pad
        self.generate_block_index() # Generate block index
        self.__generate_spatial_map()
   
    def __crop_center(self, img: np.ndarray, center_block_size: int):
        if self.__nonoverlap_blocksize == self.__overlap_blocksize:
            return img
        
        row, col = img.shape
        s_row = (row - center_block_size) // 2
        s_col = (col - center_block_size) // 2
        
        img_crop = img[s_row : s_row + center_block_size, 
                       s_col : s_col + center_block_size].copy()
        return img_crop
   
    def __padding(self, overlap_block_size=64, nonoverlap_block_size=32):
        """Pad image by specifying overlap and non-overlap sizes.
        
        Args:
            overlap_block_size (int): The big block size (e.g., 64)
            nonoverlap_block_size (int): The small block size (e.g., 32)
        """
        img = self.__img
        
        self.__nonoverlap_blocksize = nonoverlap_block_size
        self.__overlap_blocksize = overlap_block_size
        
        rows, cols = img.shape
        max_size = max(rows, cols)
        new_size = int(math.ceil(max_size / nonoverlap_block_size) * nonoverlap_block_size) + overlap_block_size
        
        pad_top = (new_size - rows) // 2
        pad_bottom = new_size - rows - pad_top
        pad_left = (new_size - cols) // 2
        pad_right = new_size - cols - pad_left
        
        self.__img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            "constant",
            constant_values=np.mean(img),
        )
        # blur mask
        if self.__blur_edge:
            blur_size, erode_size = self.__blur_edge_size
            blur_mask = np.ones_like(img)*255
            blur_mask = np.pad(
                img,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                "constant",
                constant_values=0,
            )
            self.__img = blurEdge(self.__img, blur_mask, blur_size, erode_size)
        self.__fp_pad = [pad_top, pad_bottom, pad_left, pad_right]

   
    def __unpad(self, input_img):
        """Unpad image to its original dimensions."""
        top, bottom, left, right = self.__fp_pad
        h, w = input_img.shape
        return input_img[top : h - bottom, left : w - right]
   
    def generate_block_index(self):
        o_blocksize = self.__overlap_blocksize
        no_blocksize = self.__nonoverlap_blocksize
        
        start_index = (o_blocksize - no_blocksize) // 2
        stop_index_row = self.__img.shape[0] - (o_blocksize - (start_index + no_blocksize))
        
        # ROW indices
        self.row_no_block_index_list = range(start_index, stop_index_row, no_blocksize)
        self.row_o_block_index_list = range(0, int(o_blocksize * (stop_index_row - start_index) / no_blocksize), no_blocksize)
        self.row_map_block_index_list = range(0, len(self.row_no_block_index_list) * o_blocksize, o_blocksize)
        
        # COL indices (Note: relies on stop_index_row for square processing based on original logic)
        self.col_no_block_index_list = range(start_index, stop_index_row, no_blocksize)
        self.col_o_block_index_list = range(0, int(o_blocksize * (stop_index_row - start_index) / no_blocksize), no_blocksize)
        self.col_map_block_index_list = range(0, len(self.col_no_block_index_list) * o_blocksize, o_blocksize)
   
    def __crop_map(self, map_img):
        o_blocksize = self.__overlap_blocksize
        no_blocksize = self.__nonoverlap_blocksize
        
        crop_img = np.zeros_like(self.__img, np.float32)
        
        for row_index in range(len(self.row_no_block_index_list)):
            start_no_row = self.row_no_block_index_list[row_index]
            stop_no_row = start_no_row + no_blocksize
            
            start_map_row = self.row_map_block_index_list[row_index]
            stop_map_row = start_map_row + o_blocksize
            
            for col_index in range(len(self.col_no_block_index_list)):
                start_no_col = self.col_o_block_index_list[col_index]
                stop_no_col = start_no_col + no_blocksize
                
                start_map_col = self.col_map_block_index_list[col_index]
                stop_map_col = start_map_col + o_blocksize
                
                patch = map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]
                crop_img[start_no_row:stop_no_row, start_no_col:stop_no_col] = patch
                
        return crop_img
            
    def __generate_spatial_map(self):
        o_blocksize = self.__overlap_blocksize
        
        rows_len = len(self.row_no_block_index_list)
        cols_len = len(self.col_no_block_index_list)
        
        spa_map_img = np.zeros((o_blocksize * rows_len, o_blocksize * cols_len), np.float32)
        
        for row_index in tqdm(range(rows_len)):
            start_o_row = self.row_o_block_index_list[row_index]
            stop_o_row = start_o_row + o_blocksize
            
            start_map_row = self.row_map_block_index_list[row_index]
            stop_map_row = start_map_row + o_blocksize

            for col_index in range(cols_len):
                start_o_col = self.col_o_block_index_list[col_index]
                stop_o_col = start_o_col + o_blocksize
                
                start_map_col = self.col_map_block_index_list[col_index]
                stop_map_col = start_map_col + o_blocksize
                
                patch = self.__img[start_o_row:stop_o_row, start_o_col:stop_o_col]
                spa_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col] = patch
        
        self.__spa_map_img = spa_map_img

    def stft(self):
        # Unpack Variable
        o_blocksize = self.__overlap_blocksize
        no_blocksize = self.__nonoverlap_blocksize
        spa_map_img = self.__spa_map_img
        
        rows_len = len(self.row_no_block_index_list)
        cols_len = len(self.col_no_block_index_list)
        
        freq_map_img = np.zeros((o_blocksize * rows_len, o_blocksize * cols_len), np.float32)
        phase_map_img = np.zeros((o_blocksize * rows_len, o_blocksize * cols_len), np.float32)
        
        if self.__zeromean:
            mean_arr = np.zeros((rows_len, cols_len))
            
        for row_index in tqdm(range(rows_len)):
            start_o_row = self.row_o_block_index_list[row_index]
            stop_o_row = start_o_row + o_blocksize
            
            start_map_row = self.row_map_block_index_list[row_index]
            stop_map_row = start_map_row + o_blocksize

            for col_index in range(cols_len):
                start_o_col = self.col_o_block_index_list[col_index]
                stop_o_col = start_o_col + o_blocksize
                
                start_map_col = self.col_map_block_index_list[col_index]
                stop_map_col = start_map_col + o_blocksize
                
                patch = spa_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]
                    
                FFT = Fourier2D(patch, self.__zeromean, self.__window_func)
                FFT.fft()
                
                if self.__zeromean:
                    mean_arr[row_index, col_index] = FFT.avg
                    
                freq_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col] = FFT.getMagnitude()
                phase_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col] = FFT.getPhase()
                
        self.__freq_map_img = freq_map_img
        self.__phase_map_img = phase_map_img
        if self.__zeromean:    
            self.__mean_arr = mean_arr
  
    def istft(self):
        """Invert STFT. Needs testing on real picture."""
        o_blocksize = self.__overlap_blocksize
        no_blocksize = self.__nonoverlap_blocksize
        
        output_img = np.zeros_like(self.__img, np.float32)
        
        for row_index in range(len(self.row_no_block_index_list)):
            start_no_row = self.row_no_block_index_list[row_index]
            stop_no_row = start_no_row + no_blocksize
            
            start_map_row = self.row_map_block_index_list[row_index]
            stop_map_row = start_map_row + o_blocksize
            
            for col_index in range(len(self.col_no_block_index_list)):
                start_no_col = self.col_no_block_index_list[col_index]
                stop_no_col = start_no_col + no_blocksize
                
                start_map_col = self.col_map_block_index_list[col_index]
                stop_map_col = start_map_col + o_blocksize
                
                f_patch = self.__freq_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]
                ph_patch = self.__phase_map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]
                
                FFT = Fourier2D(f_patch, frequency=True)
                FFT.setPhase(ph_patch)
                FFT.ifft()
                
                out_patch = FFT.getOutputImg()
                if self.__zeromean:
                    out_patch += self.__mean_arr[row_index, col_index]
                    
                out_patch = self.__crop_center(out_patch, no_blocksize)
                output_img[start_no_row:stop_no_row, start_no_col:stop_no_col] = out_patch
        
        # Unpad all images
        self.__img = self.__unpad(self.__img)
        self.__mask = self.__unpad(self.__mask)
        self.__freq_map_img = self.__unpad(self.__freq_map_img)
        self.__output_img = self.__unpad(output_img)
    
    def apply_func_map(self, map_img, func, *args, output_is_img=True, output_vector=None, activation_map=None, custom_row_index=None, custom_col_index=None, iteration_size=1):
        o_blocksize = self.__overlap_blocksize
        rows = len(self.row_map_block_index_list)
        cols = len(self.col_map_block_index_list)
        # If don't specify then Activate Every fucking thing.
        if activation_map is None:
            activation_map = np.full((rows, cols), True, dtype=bool)

        # Incase you don't want the output to be picture, but whatever vector you so desire
        if output_is_img:
            output_img = np.zeros((o_blocksize * rows, o_blocksize * cols), np.float32)
        else:
            output_img = output_vector
        
        row_block_index_list = self.row_map_block_index_list
        col_block_index_list = self.col_map_block_index_list
        it_size = o_blocksize
        if not output_is_img:
            if custom_row_index is None:
                custom_row_index = range(rows)
                custom_col_index = range(cols)
            row_output_block_index_list = custom_row_index
            col_output_block_index_list = custom_col_index
            opt_it_size = iteration_size
        else:
            row_output_block_index_list = row_block_index_list
            col_output_block_index_list = col_block_index_list
            opt_it_size = it_size
        for row_index in range(rows):
            # Map index
            start_map_row = row_block_index_list[row_index]
            stop_map_row = start_map_row + it_size
            # Output index, Done this for shorter computation in for loop
            o_start_map_row = row_output_block_index_list[row_index]
            o_stop_map_row = o_start_map_row+opt_it_size
            for col_index in range(cols):
                if activation_map[row_index][col_index]==True:
                    # Map index
                    start_map_col = col_block_index_list[col_index]
                    stop_map_col = start_map_col + it_size
                    # Output index
                    o_start_map_col = col_output_block_index_list[col_index]
                    o_stop_map_col = o_start_map_col+opt_it_size
                    
                    patch = map_img[start_map_row:stop_map_row, start_map_col:stop_map_col]
                    output = func(patch, *args)
                    output_img[o_start_map_row:o_stop_map_row, o_start_map_col:o_stop_map_col] = output
    
        return output_img    
        
    def get_img(self):
        return self.__img
    
    def get_mask(self):
        return self.__mask
    
    def getSpatial(self):
        return self.__spa_map_img
    
    def getMagnitude(self):
        """
        This map should not be shown in plt if o_block_size/no_block_size
        is greater than 2, otherwise the program will freeze.
        """
        return self.__freq_map_img
    
    def getPhase(self):
        return self.__phase_map_img
    
    def getVisualizeMagnitude(self):
        """Visualize the current magnitude (No overlap block)"""
        vis_magnitude = self.__crop_map(self.__freq_map_img)
        return self.__unpad(vis_magnitude)
    
    def get_output_img(self):
        return self.__output_img
    
    def setSpatial(self, spa_map):
        self.__spa_map_img = spa_map

    def setMagnitude(self, freq_map):
        self.__freq_map_img = freq_map


# if __name__ == "__main__":
#     img_path = r"D:\work\image_processing\Latent_fingerprint\SFP_latent_fingerprint_enh\segmentation\Latent_fingerprint_segmentation_KSIP2026\data\sd302h_original_500ppi\00002302_2B_X_L01_BP_S04_500PPI_8BPC_1CH_LP05-1_1.png"
#     mask_path = r"D:\work\image_processing\Latent_fingerprint\SFP_latent_fingerprint_enh\latent_fingerprint_enhancement-master\data\masks_machine\rtp2013_11_1_T_2.png"
#     img = cv.imread(img_path, 0)
#     mask = cv.imread(mask_path, 0)
#     BBF = BlockBaseFrameWork(img, mask, 64, 16, False)
#     BBF.stft()
#     magnitude = BBF.getMagnitude()
#     kurtosis = BBF.apply_func_map(magnitude, log_transform)
#     BBF.setMagnitude(kurtosis)
#     BBF.istft()
#     output = BBF.get_output_img()
#     output = normalize_range(output, (np.min(output), np.max(output)), (0, 255))

#     plot_all([img, output])