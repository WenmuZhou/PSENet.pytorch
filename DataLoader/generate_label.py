import cv2
import numpy as np
import pyclipper
def Genetate_shrinkly_poly(poly,m,n,i):
    poly = change_type(poly)
    area = cv2.contourArea(np.array(poly))
    perimeter = cv2.arcLength(np.array(poly), True)
    ri = 1 - ((1 - m) * (n - i) / (n - 1))
    di = area * (1 - ri * ri) / perimeter
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinkly_poly = np.array(pco.Execute(-di))
    return shrinkly_poly

def change_type(poly):
    poly_int=[]
    for line in poly:
        poly_int.append([int(item) for item in line])
    return poly_int
