import math
import numpy as np
from orientation_estimation import local_multipeak
from utils.check_angle_rad import check_angle_rad
'''
---- ---- ----
Under Major Reworking.
Fix Before using
-> Each find function will return multiple vector
-> Double Peak Will be Filter
-> if a peak does not have its double. Then that peak will be ignored.

---- ---- ----

'''


class BlockAttribute:
    def __init__(self, block_img, peak_pos=None):
        # ---- Declare Variable
        self.block_img = block_img
        self.row, self.col = block_img.shape
        self.x_center, self.y_center = self.col//2, self.row//2
        # ---- Initiate function
        if peak_pos is None:
            self.find_peak_pos(radius_ban=3)
    def find_peak_pos(self, radius_ban=3, max_peak_count=np.inf):
        # ---- Find peak position from local multipeak. (Better than banning peak)
        peak_pos = local_multipeak(self.block_img, radius_ban, max_peak_count)
        self.peak_pos = peak_pos
        return peak_pos
    def check_doublepeak(self):
        # ---- Check and clean double peak
        peak_pos = self.peak_pos
        row, col = self.row, self.col
        for pos in peak_pos:
            inv_pos = (row-pos[0]-1, col-pos[1]-1)
            if inv_pos in peak_pos:
                del_row = np.abs(inv_pos[0]-pos[0])
                del_col = np.abs(inv_pos[1]-pos[1])
            
    def find_direction(self, is_degree=False):
        peak_pos = self.peak_pos
        for pos in peak_pos:
            
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

    

# def find_Attribute_multi(block_img, peak_count=5, filter_double_peak=True, tol=0.8):
#     # peak_pos = local_multipeak(block_img, 3, peak_count)
#     if filter_double_peak:
#         peak_count = peak_count*2 # double peak
#     peak_pos = local_multipeak(block_img, 2, peak_count)
#     if peak_pos is None:
#         return None
#     # output_vector = np.zeros((len(peak_pos), 4))
#     output_vector = []
#     for index in range(len(peak_pos)):
#         BA = BlockAttribute(block_img, peak_pos[index])
#         direction = BA.find_direction()
#         magnitude = BA.find_magnitude()
#         frequency = BA.find_distance()
#         harmonic = BA.find_harmonic()
#         # output_vector[index] = np.array([direction, magnitude, frequency, harmonic])
#         # check for redundancy
#         if filter_double_peak:
#             is_double_peak = False
#             for vector in output_vector:
#                 if check_angle_rad(vector[0], direction) and math.isclose(vector[1], magnitude, rel_tol=tol) and math.isclose(vector[2], frequency, rel_tol=tol):
#                    is_double_peak = True
#                    break
#             if not is_double_peak:
#                 output_vector.append([peak_pos[index][0], peak_pos[index][1], direction, magnitude, frequency, harmonic])
#         else:
#             output_vector.append([peak_pos[index][0], peak_pos[index][1], direction, magnitude, frequency, harmonic])
#     return np.array(output_vector)
