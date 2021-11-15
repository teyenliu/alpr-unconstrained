import sys, os
import cv2
import numpy as np
import traceback
import argparse
import keras
import pyprind
import time

import darknet.python.darknet as dn

from os.path             import splitext, basename, isdir, isfile
from os                     import makedirs
from os.path                     import splitext, basename
from src.utils                 import crop_region, image_files_from_folder, im2single
from src.label                 import Label, lwrite, lread, Label, readShapes
from src.label                     import Shape, writeShapes
from src.keras_utils             import load_model, detect_lp
from src.drawing_utils            import draw_label, draw_losangle, write2img

from darknet.python.darknet import detect, detect_frame

from src.transform import four_point_transform
from src.transform import four_point_transform_and_replacement

# YOLO Darknet preload
vehicle_threshold = .5
vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
vehicle_dataset = 'data/vehicle-detector/voc.data'
vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
vehicle_meta = dn.load_meta(vehicle_dataset)

#WPOT Model preload
lp_threshold = .5
wpod_net_path = 'data/lp-detector/wpod-net_update1.h5'
wpod_net = load_model(wpod_net_path)

YELLOW = (  0,255,255)
RED    = (  0,  0,255)
    
if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help = "path to the video file")
    ap.add_argument("-o", "--output", help = "path to the output video file")
    ap.add_argument("-r", "--replace", help = "path to the image file for replacement")
    ap.add_argument("-c", "--coords", help = "comma seperated list of source points")
    args = vars(ap.parse_args())

    reorder_ptspx = np.ones((4,2))
    
    rep_image = cv2.imread(args["replace"])
    cap = cv2.VideoCapture(args["video"])
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    video_freq = cap.get(5)
    video_timeInterval = round(1 / cap.get(5), 3)
    video_num_frames = int(cap.get(7))

    #if video_width > 2048:
    #    video_width = int(video_width/2)
    #    video_height = int(video_height/2)

    # Using XVID encoding
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args["output"], fourcc, video_freq, (video_width, video_height))

    #while(cap.isOpened()):
    for i in pyprind.prog_bar(range(video_num_frames)):
        try:
            #ret , frame = cap.read()
            frame = cap.read()[1]
            
            if(frame is not None):
                # Using new Darknet's detect function for adopting numpy array as image input
                R,_ = detect_frame(vehicle_net, vehicle_meta, frame ,thresh=vehicle_threshold)
                R = [r for r in R if r[0] in ['car','bus']]
                #print '\t\t%d cars found' % len(R)
                if len(R):
                    Iorig = frame.copy()
                    WH = np.array(Iorig.shape[1::-1],dtype=float)
                    #Lcars = []
                    for i,r in enumerate(R):
                        cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                        tl = np.array([cx - w/2., cy - h/2.])
                        br = np.array([cx + w/2., cy + h/2.])
                        label = Label(0,tl,br)
                        Ivehicle = crop_region(Iorig,label)
                        #Draw car label
                        draw_label(frame,label,color=YELLOW,thickness=1)
                        #Lcars.append(label)
                        
                        # Detect license plate
                        ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
                        side  = int(ratio*288.)
                        bound_dim = min(side + (side%(2**4)),608)
                        #print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)

                        Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
                        if len(LlpImgs):
                            #print '\t\t%d license plate found' % len(LlpImgs)
                            #Ilp = LlpImgs[0]
                            #Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                            #Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                            s = Shape(Llp[0].pts)
                            pts = s.pts*label.wh().reshape(2,1) + label.tl().reshape(2,1)
                            ptspx = pts*np.array(Iorig.shape[1::-1],dtype=float).reshape(2,1)

                            #draw_losangle(frame,ptspx,RED,3)
                            #Danny Modification for drawing messages on license plate
                            for i in range(4):
                                for j in range(2):
                                    reorder_ptspx[i][j]=ptspx[j][i]
                            reverse_warped = four_point_transform_and_replacement(frame, reorder_ptspx, rep_image)

                #save the result into video output file
                #cv2.imshow('frame', frame)
                out.write(frame)
            else:
                print("An empty frame...\n")

            if cv2.waitKey(1) &0xFF ==ord('q'):  #press key:q to quit
                break
        except:
            traceback.print_exc()
            sys.exit(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    

