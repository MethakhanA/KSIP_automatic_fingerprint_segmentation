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
        self.x_center, self.y_center = self.col/2, self.row/2
        # ---- Initiate function
        if peak_pos is None:
            self.find_peak_pos(radius_ban=3)
        self.filter_double_peak() # --- Check for double peak
        self.find_direction(is_degree=False) # --- Check for direction
        self.find_magnitude() # --- Find magnitude
        self.find_distance() # --- Find Frequency or Distance
        self.find_direction() # --- Find if there is Harmonic
        # ----
    def find_peak_pos(self, radius_ban=3, max_peak_count=np.inf):
        # ---- Find peak position from local multipeak. (Better than banning peak)
        peak_pos = local_multipeak(self.block_img, radius_ban, max_peak_count)
        self.peak_pos = peak_pos
        return peak_pos
    def filter_double_peak(self):
        # ---- Couple Double peak together
        peak_pos = self.peak_pos
        row, col = self.row, self.col
        couple_peak = {}
        for pos in peak_pos:
            inv_pos = (row-pos[0]-1, col-pos[1]-1)
            if inv_pos in peak_pos:
                couple_peak[(pos, inv_pos)] = {}
        self.couple_peak = couple_peak
        return couple_peak
    def find_direction(self, is_degree=False):
        # ---- Find all direction inside the block
        couple_peak = self.couple_peak
        for (pos, inv_pos) in couple_peak:
            del_row = np.abs(pos[0]-inv_pos[0])
            del_col = np.abs(pos[1]-inv_pos[1])
            angle = math.atan2(del_row, del_col)
            if pos[0]<inv_pos[0]:
                angle += np.pi/2
            if is_degree:
                angle = math.degrees(angle)
            # couple_peak.append([(pos, inv_pos), angle])
            couple_peak[(pos, inv_pos)]["angle"] = angle        
        # ---- Output is an ndarray in the from of
        # [[[pos, inv_pos], angle], ...]
        self.couple_peak = couple_peak
        return couple_peak

    def find_magnitude(self):
        # ---- Pack magnitude into array
        couple_peak = self.couple_peak
        block_img = self.block_img
        for (pos, inv_pos) in couple_peak:
            couple_peak[(pos, inv_pos)]["magnitude"] = block_img[pos[0], pos[1]]
        self.couple_peak = couple_peak
        return couple_peak

    def find_distance(self):
        # ---- find distance from center
        x_center, y_center = self.x_center, self.y_center
        couple_peak = self.couple_peak
        for (pos, inv_pos) in couple_peak:
            d_y = pos[0]-y_center
            d_x = pos[1]-x_center
            d_r = np.sqrt(d_x**2+d_y**2)
            couple_peak[(pos, inv_pos)]["distance"] = d_r
        self.couple_peak = couple_peak
        return couple_peak

    def find_harmonic(self):
        # ---- find if there is harmonic
        row, col = self.row, self.col # --- Unpack Variable
        couple_peak = self.couple_peak
        block_img = self.block_img
        for (pos, inv_pos) in couple_peak:
            angle = couple_peak[(pos, inv_pos)]["angle"]
            h_y, h_x = int(2*pos[0]*np.sin(angle)), int(2*pos[1]*np.cos(angle))
            if h_y>=row or h_x>=col:
                harmonic = None
            else:
                harmonic = block_img[h_y, h_x]
            couple_peak[(pos, inv_pos)]["harmonic"] = harmonic # --- assign value
        self.couple_peak = couple_peak
        return couple_peak

    def getattribute(self):
        return self.couple_peak



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
