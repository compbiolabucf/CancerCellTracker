#!/usr/bin/env python3
import cv2
import os
import numpy as np
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def read_frame(path, frame_count, data_type, scale, crop_width = 0, crop_height = 0):
    # global crop_width, crop_height

    if (data_type == 0):#read from raw image data
        video_full_path = path + "input_images/"
        image_path = video_full_path + str(frame_count) + ".npy" #In default case, npy files have been scaled 8 times.
        ret = os.path.exists(image_path)
        if ret != True:
            print("File not exists, ", image_path)
            return False, None

        frame = np.load(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = frame[0:crop_height * scale, 0:crop_width * scale]
        return True, frame
    elif (data_type == 1 or data_type == 2):  # read from jpg png tiff
        image_path = path + "t" + "{0:0=3d}".format(frame_count) + ".tif"
        if ((not os.path.exists(image_path))): #  or frame_count > 30
            print("file not exist: ", image_path)
            return False, None
        else:
            # print(frame_count, image_path)
            pass

        # frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # x = 619
        # y = 321
        # frame = frame[y:y + crop_height, x:x + crop_width]


        if(crop_width == 0 and crop_height == 0):
            crop_width = frame.shape[1]
            crop_height = frame.shape[0]


        frame = frame[0:crop_height, 0:crop_width]
        if (scale > 1):
            frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        return True, frame
