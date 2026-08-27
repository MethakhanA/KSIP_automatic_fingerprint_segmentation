import math
import numpy as np
from orientation_estimation import local_multipeak


class BlockAttribute:
    def __init__(self, block_img, peak_pos):
        self.block_img = block_img
        self.peak_pos = peak_pos
        # 
        self.row, self.col = block_img.shape
        self.x_center, self.y_center = self.col//2, self.row//2
    def find_direction(self, is_degree=False):
        x_center, y_center = self.x_center, self.y_center
        # find angle from 0
        peak_y, peak_x = self.peak_pos
        d_y = peak_y-y_center
        d_x = peak_x-x_center
        angle = math.atan2(d_y, d_x)
        if is_degree:
            # convert to degree if wanted
            angle = math.degrees(angle)
        self.angle = angle
        return angle
    def find_magnitude(self):
        # Unpack Variable
        peak_y, peak_x = self.peak_pos
        magnitude = self.block_img[peak_y, peak_x]
        return magnitude
        
    def find_distance(self):
        # find distance from center
        x_center, y_center = self.x_center, self.y_center
        peak_y, peak_x = self.peak_pos
        d_y = peak_y-y_center
        d_x = peak_x-x_center
        d_r = np.sqrt(d_x**2+d_y**2)
        self.distance = d_r
        return d_r
    def find_harmonic(self):
        # check if there is anything at double frequency
        peak_y, peak_x = self.peak_pos
        row, col = self.row, self.col
        angle = self.angle
        h_y, h_x = int(2*peak_y*np.sin(angle)), int(2*peak_x*np.cos(angle))
        if h_y>=row or h_x>=col:
            return None
        return self.block_img[h_y, h_x]
def find_Attribute_multi(block_img):
    peak_pos = local_multipeak(block_img, 3, 10)
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
