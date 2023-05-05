import time
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolovdetect import (yolov_object_detect, yolov_object_n_action_detect)
from action_model import (test_live_video, test_video, test_annots, test_random_live_video, test_random_video, load_virat_dataset, 
    random_video, generate_yolov_object_annotations, test_live_video_yolov_annotation, split_video_yolov, split_all_yolov_videos, 
    generate_yolov_action_annotations,split_all_yolov_action_videos, create_tublet_videos, split_test_train_val)

'''This file is a suite with the ability to run different
    parts of the system'''

    #
    #### These functions are used for testing and manipulating the datset
name = "VIRAT_S_000006"
#test_live_video(name)
test_video(name)
#test_annots()

#test_random_live_video()
#test_random_video()

#load_virat_dataset()

#generate_yolov_object_annotations()
#split_all_yolov_videos()

#generate_yolov_action_annotations()

#split_all_yolov_action_videos()

#create_tublet_videos()

#split_test_train_val()

#test_live_video_yolov_annotation(random_video())
    ####
  #
#
  #
    #### These functions are to run the object and action detection
PATH = "C:\\Users\\hugo\\Documents\\University\\YEAR_3\\ECM3401_Project\\Action Detection\\datasets\\VIRAT\\Public Dataset\\VIRAT Video Dataset Release 2.0\\VIRAT Ground Dataset\\videos_original\\"

#source = PATH+random_video()+".mp4"
#source = PATH+"VIRAT_S_000200_01_000226_000268"+".mp4"
#source = PATH+"VIRAT_S_000200_00_000100_000171"+".mp4"
#source = PATH+"VIRAT_S_040104_04_000854_000934"+".mp4"
source = PATH+"VIRAT_S_000006"+".mp4"
#source = PATH+"VIRAT_S_000203_08_001702_001734"+".mp4"

    #All the object/action detection parameters
weights              = "yolov7.pt"
save_video           = True
view_video           = False
save_txt             = False
imgsz                = 640
trace                = False
show_tracking        = False
skip_rate            = 3     #The object detection skip rate
draw_action_boxes    = True
draw_object_boxes    = False
draw_proposal_boxes  = False
show_foregroundseg   = False
refinement_threshold = 5

print("Video chosen: ")
print(source+"\n")

    ##To just run the object detection
#yolov_object_detect(source, weights, save_video, view_video, save_txt, imgsz, trace)

    #To run the action detection, or show any other aspect of
    #the system as described in the paramaeters
yolov_object_n_action_detect(source, weights, save_video, view_video, save_txt, imgsz, trace, show_tracking, skip_rate, draw_action_boxes, draw_object_boxes, draw_proposal_boxes, show_foregroundseg, refinement_threshold)
