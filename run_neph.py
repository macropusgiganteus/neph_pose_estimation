import argparse
import logging
import time
from xlwt import Workbook 


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
import scipy.signal as signal
from scipy.interpolate import make_interp_spline, BSpline
from matplotlib.pyplot import plot, scatter, show
from sklearn.metrics import mean_squared_error 

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
workbook = Workbook()
position = workbook.add_sheet("Part_Position")
position.write(0, 0, 'body part') 
position.write(0, 1, 'x') 
position.write(0, 2, 'y') 
position.write(0, 3, 'frame') 
whole_pos = []

import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array



def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 

        return ang_deg
    
body_part = {
  '0':  "Nose",
  '1':  "Neck", # squat
  '2':  "RShoulder", # bicepcurl
  '3':  "RElbow", # bicepcurl
  '4':  "RWrist", # bicepcurl
  '5':  "LShoulder",
  '6':  "LElbow",
  '7':  "LWrist",
  '8':  "MidHip", 
  '9':  "RHip",
  '10': "RKnee",
  '11': "RAnkle",
  '12': "LHip", # squat
  '13': "LKnee", # squat
  '14': "LAnkle", # squat
  '15': "REye",
  '16': "LEye",
  '17': "REar",
  '18': "LEar",
  '19': "LBigToe",
  '20': "LSmallToe",
  '21': "LHeel",
  '22': "RBigToe",
  '23': "RSmallToe",
  '24': "RHeel",
  '25': "Background"
}



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

all_angle = []
angle_frame = []
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--resize', type=str, default='656x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--vertical', type=bool, default=False)
    parser.add_argument('--flip', type=bool, default=False)
    parser.add_argument('--movement', type=str, default='squat', help='type of exercise')
    parser.add_argument('--plot', type=str, default='none', help='plot name')

    parser.add_argument('--record', type=str, default='none', help='record file name')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    w, h = model_wh(args.resize)
    print(w,h)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        dsize = (w,h)
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')


    if args.video != '':
        cam = cv2.VideoCapture(args.video)
    else:
        cam = cv2.VideoCapture(args.camera)

    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    if cam.isOpened() is False:
        print("Error opening video stream or file") 


    part = 1
    frame = 1
    while cam.isOpened():
        
        ret_val, image = cam.read()
        if w > 0 and h > 0:
            try:
                image = cv2.resize(image, dsize,cv2.INTER_AREA)
            except:
                break
        if args.vertical == True:
            image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)

        if args.flip == True:
            image = cv2.flip(image, 1)

        if not ret_val or cv2.waitKey(1) == 27:
            break   
        
            
        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        if not args.showBG:
            image = np.zeros(image.shape)


        logger.debug('postprocess+')
        image ,all_part = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        whole_pos.append(all_part)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        
        Rwrist_pos = []
        Relbow_pos = []
        Rshoulder_pos = []
        neck_pos = []
        Lhip_pos = []
        Lknee_pos = []
        Langle_pos = []
        have_shoulder = False
        have_wrist = False
        have_Lhip = False
        have_neck = False
        for p in all_part: 
            part_s = str(p).find(':')
            part_e = str(p).find('-')
            body_part_index = str(p)[part_s+1:part_e]

            x_s = str(p).find('(')
            x_e = str(p).find(',')
            x_pos = str(p)[x_s+1:x_e]

            y_s = str(p).find(',')
            y_e = str(p).find(')')
            y_pos =  str(p)[y_s+1:y_e]

            if args.movement.upper() == 'BICEPCURL':
                Bicep_joint = ['RShoulder','RElbow','RWrist']
                if str(body_part[body_part_index]) in Bicep_joint :
                    part = part +1
                    have_shoulder = True
                    have_wrist = True
                    if str(body_part[body_part_index]) == 'RWrist' : Rwrist_pos = [float(x_pos),float(y_pos)]
                    elif str(body_part[body_part_index]) == 'RShoulder' : Rshoulder_pos = [float(x_pos),float(y_pos)]
                    elif str(body_part[body_part_index]) == 'RElbow' : Relbow_pos = [float(x_pos),float(y_pos)]
                    cv2.putText(image,
                                str(body_part[body_part_index])+' ('+x_pos+','+y_pos+')',
                                (10, part*15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
                
            elif args.movement.upper() == 'SQUAT':
                Squat_joint = ['Neck','LHip','LKnee','LAnkle']
                if str(body_part[body_part_index]) in Squat_joint :
                    part = part +1
                    have_Lhip = True
                    have_neck = True
                    if str(body_part[body_part_index]) == 'Neck' : neck_pos = [float(x_pos),float(y_pos)]
                    elif str(body_part[body_part_index]) == 'LHip': Lhip_pos = [float(x_pos),float(y_pos)]
                    elif str(body_part[body_part_index]) == 'LKnee' : Lknee_pos = [float(x_pos),float(y_pos)]
                    elif str(body_part[body_part_index]) == 'LAnkle' : Langle_pos = [float(x_pos),float(y_pos)]
                    cv2.putText(image,
                                str(body_part[body_part_index])+' ('+x_pos+','+y_pos+')',
                                (10, part*15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
        
        if args.movement.upper() == 'BICEPCURL':
            upper_arm = [Relbow_pos,Rshoulder_pos]
            lower_arm = [Relbow_pos,Rwrist_pos]
            print(upper_arm)
            print(lower_arm)
            angle_in_frame = []   
            try :
                if have_shoulder and have_wrist:
                    angle = ang(upper_arm , lower_arm)
                    part = part + 1
                    cv2.putText(image,
                                        'angle = '+str(angle),
                                        (10, part*15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)
                    all_angle.append(angle)
                    angle_frame.append(frame)
            except:
                cv2.putText(image,
                                        'can not find angle',
                                        (10, part*15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)

        elif args.movement.upper() == 'SQUAT':
            upper_Lleg = [Lhip_pos,Lknee_pos]
            back = [Lhip_pos,neck_pos]
            print(upper_Lleg)
            print(back)
            angle_in_frame = []   
            try :
                if have_Lhip and have_neck:
                    angle = ang(upper_Lleg , back)
                    part = part + 1
                    cv2.putText(image,
                                        'angle = '+str(angle),
                                        (10, part*15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)
                    all_angle.append(angle)
                    angle_frame.append(frame)
            except:
                cv2.putText(image,
                                        'can not find angle',
                                        (10, part*15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)



        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        part = 1
        frame += 1
    
    all_angle = np.array(all_angle)
    angle_frame = np.array(angle_frame)
    angle_frame_new = np.linspace(angle_frame.min(), angle_frame.max(), 200) 
    #define spline
    spl = make_interp_spline(angle_frame, all_angle, k=3)
    all_angle_smooth = spl(angle_frame_new)

    peaks = find_peaks_cwt(all_angle_smooth, np.arange(10,50))
    # minima = np.r_[True, all_angle_smooth[1:] < all_angle_smooth[:-1]] & np.r_[all_angle_smooth[:-1] < all_angle_smooth[1:], True]
    # minima = signal.argrelextrema(all_angle_smooth, np.less)
    minima = np.where((all_angle_smooth[1:-1] < all_angle_smooth[0:-2]) * (all_angle_smooth[1:-1] < all_angle_smooth[2:]))[0] + 1
    new_minima = []
    for m in minima: 
        if all_angle_smooth[m] < 80 : 
            if len(new_minima) == 0 : new_minima.append(m)
            elif m - new_minima[-1] > 5 : 
                if m - new_minima[-1] < 15 and all_angle_smooth[m] < all_angle_smooth[-1]:
                    new_minima[-1] = m
                else: new_minima.append(m)
                

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(angle_frame_new, all_angle_smooth, label='angle')  # Plot some data on the axes.
    ax.scatter(angle_frame_new[peaks], all_angle_smooth[peaks], color='r')
    ax.scatter(angle_frame_new[new_minima], all_angle_smooth[new_minima], color='b')
    # upper_treshold = [175]*len(angle_frame)
    # lower_treshold = [20]*len(angle_frame)
    
    # ax.plot(angle_frame_new, upper_treshold, label='upper_treshold')
    # ax.plot(angle_frame_new, lower_treshold, label='lower_treshold')
    
    print(new_minima)
    print(peaks)
    print('local maxima',all_angle_smooth[peaks])
    print('local minima',all_angle_smooth[new_minima])
    maxima_true = [175]*len(all_angle_smooth[peaks])
    minima_true = [20]*len(all_angle_smooth[new_minima])
    maxima_error = mean_squared_error(maxima_true,all_angle_smooth[peaks])
    minima_error = mean_squared_error(minima_true,all_angle_smooth[new_minima])
    print('maxima error =',maxima_error)
    print('minima error =',minima_error)

    if(maxima_error > 20): print('spread arm wider!')
    if(minima_error > 20): print('pull wrist closer to body when you lift!')
    if maxima_error < 20 and minima_error < 20 : print('Good job! you did it right!')

    ax.set_xlabel('frame')  # Add an x-label to the axes.
    ax.set_ylabel('angle')  # Add a y-label to the axes.
    ax.set_title("Right elbow angle")  # Add a title to the axes.
    ax.legend()
    if args.plot != 'none':
        plt.savefig(args.plot+'.png')
    cv2.destroyAllWindows()
logger.debug('finished+')



# row = 0
# output = []
# for frame in range(len(whole_pos)):
#     for body_p in range(len(whole_pos[frame])):
#         output_s = []

#         part_s = str(whole_pos[frame][body_p]).find(':')
#         part_e = str(whole_pos[frame][body_p]).find('-')
#         body_part_index = str(whole_pos[frame][body_p])[part_s+1:part_e]

#         x_s = str(whole_pos[frame][body_p]).find('(')
#         x_e = str(whole_pos[frame][body_p]).find(',')
#         x_pos = str(whole_pos[frame][body_p])[x_s+1:x_e]

#         y_s = str(whole_pos[frame][body_p]).find(',')
#         y_e = str(whole_pos[frame][body_p]).find(')')
#         y_pos =  str(whole_pos[frame][body_p])[y_s+1:y_e]
#         output_s.append(body_part[body_part_index])
#         output_s.append(x_pos)
#         output_s.append(y_pos)
#         output_s.append(frame+1)

#         position.write(row+1, 0, str(body_part[body_part_index])) 
#         position.write(row+1, 1, str(x_pos)) 
#         position.write(row+1, 2, str(y_pos)) 
#         position.write(row+1, 3, frame+1)
#         row += 1
#         output.append(output_s)

# workbook.save(args.record+'.xls')  