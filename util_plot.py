import cv2
import numpy as np
def show_segs(img,segs,save_dir=None):
    for seg in segs:
        print(seg)
        pts = np.array(seg, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,255,255))     

    if save_dir:
        pass
    else:
        cv2.imshow('image', img)
        cv2.waitKey(0)