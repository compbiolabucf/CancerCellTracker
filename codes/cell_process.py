#!/usr/bin/env python3
import cv2
from cell_detect import CellDetector
from cell_classify import CellClassifier
import os
import numpy as np
import time
import sys
import re
from util import read_frame

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# home_dir = os.path.expanduser("~") + "/"

debug = 0

crop_width = 0
crop_height = 0

# crop_width = 1328
# crop_height = 1048

# crop_width = 512
# crop_height = 512

# crop_width = 256
# crop_height = 256

# crop_width = 730
# crop_height = 1024


scale = 8

line_thick = 1
debug = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')


def main(configure_path = "./configure.txt", path = "Work/ground_truth/preprocess", out_path = "Default"):

    path = os.path.abspath(path)
    out_path = os.path.abspath(out_path)

    if(out_path == "Default"):
        out_path = re.sub(r'RawData.*', 'TimeLapseVideos/', path)
        print("Output directory is: ", out_path)
    # else:
    #     out_path = home_dir + out_path

    ret = re.sub(r'.*Beacon-', '', path)
    Beacon = re.sub(r'/.*', '', ret)

    if(Beacon == ''):
        Beacon = 0
    else:
        Beacon = int(Beacon)

    paras = []
    with open(configure_path) as f:
        for l in f:
            l = re.sub(r'#.*', '', l)# replace the comments with ''
            l = l.replace(" ", "")
            l = l.replace("\n", "")
            if(len(l) > 0):
                paras.append(l.split("="))

    paras_dict = {p[0]:p[1] for p in paras}

    for key in ["cell_core_radius_range_2", "cell_core_radius_range_3"]:
        radius_interval = paras_dict[key]
        radius_interval = radius_interval.replace("(", "")
        radius_interval = radius_interval.replace(")", "")
        radius_interval = radius_interval.split(",")
        radius_interval = [float(radius_interval[0]), float(radius_interval[1])]
        paras_dict[key] = radius_interval

    for key in ["cell_max_1", "black_edge_2", "white_core_2", "white_core_3", "cell_max_3"]:
        paras_dict[key] = float(paras_dict[key])

    print("Mode: ", paras_dict["Mode"])

    if (path[-1] != '/'):
        path += '/'

    if (out_path[-1] != '/'):
        out_path += '/'

    os.makedirs(out_path, exist_ok = True)

    for new_folder in ["images_ucf/Beacon_" + str(Beacon) + "/", "videos_ucf/", "Results_ucf/", "info_ucf/", "Cell_tracks/", "Results/"]:
        os.makedirs(out_path + new_folder, exist_ok = True)


    process_one_video_main(path, Beacon, 1, None, out_path, paras_dict)

def process_one_video_main(path, Beacon, data_type, pt, out_path, paras_dict):
    global crop_width, crop_height
    t0 = time.time()
    t0_str = time.ctime(t0)

    detector = CellDetector()
    classifier = CellClassifier(8, 20, 5, 0)
    # ph_detector = PhagocytosisDetector(10, 30, 5, 0)
    image_path = path
    develop = 0
    if(develop == 0):
        detector.prepro_frames_2(image_path, out_path + "images_ucf/Beacon_" + str(Beacon) + "/")
        classifier.image_amount = detector.image_amount

        if(paras_dict["Mode"] == '2'):
            detector.edge_thr = paras_dict["black_edge_2"]
            detector.core_thr = paras_dict["white_core_2"]
            detector.radius_thr = paras_dict["cell_core_radius_range_2"]
            pass
        elif (paras_dict["Mode"] == '3'):
            detector.core_thr = paras_dict["white_core_3"]
            detector.radius_thr = paras_dict["cell_core_radius_range_3"]
            detector.max_pixel = paras_dict["cell_max_3"]
            pass
        else:#this is mode 0 and 1

            if (paras_dict["Mode"] == '1'):
                detector.max_pixel = paras_dict["cell_max_1"]
                pass
            # detector.max_pixel = 200
    else:
        detector.background_pixel = 100
        detector.edge_thr = detector.background_pixel_mean = 99.57
        detector.background_pixel_std = 18.02
        detector.bg_gau_mean = 100.24
        detector.bg_gau_std = 4.67
        detector.cell_core_r = 2.88
        detector.cell_core_r_std = 0.46
        detector.noise_radius_thresh = 1.0
        detector.core_thr = 120

        detector.radius_thr = [1, 10]
        detector.image_amount = 10000
        paras_dict["Mode"] = '3'
        print("Developing Mode using Detection Mode: ", paras_dict["Mode"])

    det_out = None
    tra_out = None
    # make_video = True
    make_video = False

    frame_prev = None
    image_amount_str = str(detector.image_amount)
    print("detect and track:")
    for frame_count in range(detector.image_amount):
        # print(str(Beacon) + "_" + str(frame_count), end = " ", flush=True)
        print("\r", frame_count, end = "/" + image_amount_str, flush=True)
        ret, frame_org = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/", frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)

        if(crop_width == 0 and crop_height == 0):
            crop_width = int(frame_org.shape[1]/scale)
            crop_height = int(frame_org.shape[0]/scale)


        if(ret == False):
            detector.image_amount = frame_count
            print("done")
            break

        frame_det = frame_org.copy()

        if(paras_dict['Mode'] == '0' or paras_dict['Mode'] == '2'):
            # frame_det, centers = detector.detect_by_edge_core_and_level_RFP(out_path, frame_det, frame_count, scale)
            frame_det, centers = detector.detect_by_edge_core_and_level(out_path, frame_det, frame_count, scale)
        elif(paras_dict['Mode'] == '1' or paras_dict['Mode'] == '3'):
            frame_det, centers = detector.detect_by_white_core_and_level(frame_det, frame_count, scale)
        else:
            print("Mode is not defined: ", paras_dict['Mode'])
            pass

        cell_count = 0
        if len(centers) > 0:
            frame_tra = frame_org.copy()

            frame_tra = classifier.match_track_3_times(centers, frame_prev, frame_tra, frame_count, scale)

            for i in range(len(centers)):
                # print(len(arr), arr[:, 3].sum(), end=" ")
                cell_count = cell_count + len(centers[i])

            pass
        else:
            print("not detected")


        if make_video == True:
            if det_out is None:
                # det_out = cv2.VideoWriter(out_path + "cell_detect_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 3.0, (frame_det.shape[1], frame_det.shape[0]), isColor=True)
                det_out = cv2.VideoWriter(out_path + "cell_detect.mp4", fourcc, 3.0, (frame_det.shape[1], frame_det.shape[0]), isColor=True)
            if tra_out is None:
                # tra_out = cv2.VideoWriter(out_path + "cell_track_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 3.0, (frame_tra.shape[1], frame_tra.shape[0]), isColor=True)
                tra_out = cv2.VideoWriter(out_path + "cell_track.mp4", fourcc, 3.0, (frame_tra.shape[1], frame_tra.shape[0]), isColor=True)

        if (det_out != None):
            det_out.write(frame_det)
        if (tra_out != None):
            tra_out.write(frame_tra)


        frame_prev = frame_org.copy()
    print()


    # print("Done!")

    if (det_out != None):
        det_out.release()

    if (tra_out != None):
        tra_out.release()

    classifier.background_pixel = detector.background_pixel
    classifier.cell_core_r = detector.cell_core_r
    classifier.cell_core_r_mean = detector.cell_core_r_mean


    # classifier.analyse_classification_3(image_path, frame_count)
    gt_video_path = ""
    gt_video_path = re.sub(r'RawData.*', 'TimeLapseVideos/', path) + "Beacon-" + str(Beacon) + "processed.avi"
    # gt_video_path = home_dir + "Work/ground_truth/RFP.mp4"
    # classifier.analyse_classification_7(out_path, detector.image_amount, gt_video_path, scale, Beacon)

    gt = False
    classifier.analyse_classification_8_win(out_path, detector.image_amount, gt_video_path, scale, Beacon, gt)
    
    mark_ground_truth(classifier, out_path + "images_ucf/Beacon_" + str(Beacon) + "/", Beacon, data_type, 8, detector.image_amount, out_path, gt_video_path)

    # mark(classifier, path, Beacon, data_type, scale)

    cv2.destroyAllWindows()

def mark_ground_truth(worker, image_path, Beacon, data_type, scale, frame_amount, out_path, gt_video_path):

    out2 = None
    vid = None
    frame_count = 0

    # gt = False
    gt = True
    save_img = False
    # get_cells = True
    get_cells = False
    f_det_txt = None

    if(gt == True and os.path.exists(gt_video_path)):
        # print(out_path + "Beacon-" + str(Beacon) + "processed.avi")
        vid = cv2.VideoCapture(gt_video_path)
        if(vid):
            skip = frame_amount - int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            print("mark cells skip: ", skip)

    print("mark cells:")
    image_amount_str = str(frame_amount)

    if (not os.path.exists(out_path + "tmp/" + "Beacon-" + str(Beacon) + "/")):
        os.makedirs(out_path + "tmp/" + "Beacon-" + str(Beacon) + "/")

    if(get_cells):
        os.makedirs(out_path + "/ML/", exist_ok=True)
        os.makedirs(out_path + "/ML/images/", exist_ok=True)
        cells_path = out_path + "/ML/cells/Beacon_" + str(Beacon) + "/"
        os.makedirs(cells_path, exist_ok=True)
        f_det_txt = open(out_path + "/ML/det.txt", "w")

    for frame_count in range(frame_amount):
        ret, frame = read_frame(image_path, frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)
        if(ret == False):
            print("done")
            return

        print("\r", frame_count, end="/" + image_amount_str, flush=True)

        if(len(frame.shape) == 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        gt_frame = np.zeros(0)
        ret = False
        if(gt and vid and frame_count >= skip):
            ret, gt_frame = vid.read()

        # frame = worker.mark(frame, frame_count, scale)
        # ret_1, frame_1 = read_frame(image_path, Beacon, frame_count, data_type, 1)

        # print("gt and ret", gt, ret)


        frame, frame_red = worker.mark_gt(frame, frame_count, scale, gt_frame, crop_height, crop_width, out_path, Beacon, gt and ret, get_cells, f_det_txt)
        # ph_detector.mark_cells(frame, frame_count)

        size = (crop_width * 3, crop_height * 3)
        # size = (crop_width, crop_height)
        if(out2 is None):
            # out2 = cv2.VideoWriter(out_path + "cell_classified_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4",fourcc, 3.0, (crop_width * 3, crop_height * 3), isColor=True)
            out2 = cv2.VideoWriter(out_path + "videos_ucf/Beacon-" + str(Beacon) + "-classified.mp4",fourcc, 3.0, size, isColor=True)

        if out2:
            if(save_img):
                cv2.imwrite(out_path + "tmp/Beacon-" + str(Beacon) + "/mark_img_" + str(frame_count) + ".png", frame)

            frame_vid = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
            out2.write(frame_vid)

            if(gt and ret and save_img):
                frame_red = cv2.resize(frame_red, (crop_width * 8, crop_height * 8), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(out_path + "tmp/Beacon-" + str(Beacon) + "/imj_j" + str(frame_count) + ".png", frame_red)

    if(get_cells):
        f_det_txt.close()

    # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Tracking', 900,900)
    # cv2.imshow('Tracking', frame)
    # cv2.waitKey()

    print("\n")

    print("Done!")
    # cap2.release()
    if(out2):
        out2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if(len(sys.argv[1:]) > 1):
        main(*sys.argv[1:])
    else:
        home_dir = os.path.expanduser("~") + "/"
        main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt204/RawData/Beacon-73", out_path = "Default")

