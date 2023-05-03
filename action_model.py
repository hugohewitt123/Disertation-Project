import os
import cv2
import sys
import json
import time
import random
import datetime
import numpy as np
import skimage.draw
import pandas as pd
import tensorflow as tf
from pathlib import Path
import tensorflow.keras as keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.backend as K

event_types =  ["na", 
                "Person loading an Object to a Vehicle", 
                "Person Unloading an Object from a Car/Vehicle",
                "Person Opening a Vehicle/Car Trunk",
                "Person Closing a Vehicle/Car Trunk",
                "Person getting into a Vehicle",
                "Person getting out of a Vehicle",
                "Person gesturing",
                "Person digging",
                "Person carrying an object",
                "Person running",
                "Person entering a facility",
                "Person exiting a facility"]

object_types = ["na",
                "person",
                "car",
                "vehicles",
                "object",
                "bike, bicylces"]
#1: person
#2: car              (usually passenger vehicles such as sedan, truck)
#3: vehicles         (vehicles other than usual passenger cars. Examples include construction vehicles)
#4: object           (neither car or person, usually carried objects)
#5: bike, bicylces   (may include engine-powered auto-bikes)


#Path to dataset
PATH = ""

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

event_colors = random_colors(len(event_types))
event_dict = {
    name: color for name, color in zip(event_types, event_colors)
}

object_colors = random_colors(len(object_types))
object_dict = {
    name: color for name, color in zip(object_types, object_colors)
}

def read_annots_object_only(annot_object):
    '''Function to read and decode the annotations for the VIRAT dataset'''
    objects = pd.read_csv(annot_object, sep=" ", header=None)
    return objects

def read_annots_events_only(annot_event):
    '''Function to read and decode the annotations for the VIRAT dataset'''
    events = pd.read_csv(annot_event, sep=" ", header=None, usecols = [0,1,2,3,4,5,6,7,8,9])
    return events
    
def read_annots(annot_event, annot_map, annot_object):
    '''Function to read and decode the annotations for the VIRAT dataset'''
    events = pd.read_csv(annot_event, sep=" ", header=None, usecols = [0,1,2,3,4,5,6,7,8,9])
    objects = pd.read_csv(annot_object, sep=" ", header=None)
    maps = pd.read_csv(annot_map, sep=" ", header=None, usecols = [0,1,2,3,4,5])
    assoc_maps = pd.read_csv(annot_map, sep=" ", header=None)
    assoc_maps = assoc_maps[assoc_maps.columns[7:]]

    return events, objects, maps, assoc_maps

def display_events(frame, events):
    '''Function to display all the events in a video'''
    colour = event_dict[event_types[events[1]]]
    x1 = events[6]
    y1 = events[7]
    x2 = events[6]+events[8]
    y2 = events[7]+events[9]

    image = cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    image = cv2.putText(
        image, event_types[events[1]], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, colour, 2
        )
    return image

def display_objects(frame, objects):
    '''Function to display all the objects in a video'''
    colour = object_dict[object_types[objects[7]]]
    x1 = objects[3]
    y1 = objects[4]
    x2 = objects[3]+objects[5]
    y2 = objects[4]+objects[6]

    image = cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    image = cv2.putText(
        image, object_types[objects[7]], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, colour, 2
        )
    return image

def test_video(name):
    '''A fucntion to test video annotations'''
    path1 = PATH + "videos_original\\"
    stream = cv2.VideoCapture(path1+name+".mp4")
    width, height = stream.get(cv2.CAP_PROP_FRAME_WIDTH), stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('outtest.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))
    count = 0
    try:
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        print("error getting total frames")
        total_frames = "NaN"

    path = PATH + "annotations\\"
    fails = []
    objectonly = False
    try:
        events, objects, maps, assoc_maps = read_annots(path+name+".viratdata.events.txt", path+name+".viratdata.mapping.txt", path+name+".viratdata.objects.txt")
    except:
        try:
            objects = read_annots_object_only(path+name+".viratdata.objects.txt")
            objectonly = True
        except:
            fails.append(path+name+"does not exist, possible there are no actions in the video")
    if fails:
        print(fails)
        return
    start_time = time.time()
    # FPS update time in seconds
    display_time = 1
    fc=0
    FPS = 0
    frames = []
    while True:
        ret, frame = stream.read()
        count += 1
        fc+=1
        TIME = time.time() - start_time

        if not objectonly:
            count_events = events.loc[events[5] == count]
            for index, row in count_events.iterrows():
                if row[1] != 0:
                    frame = display_events(frame, row)

        count_objects = objects.loc[objects[2] == count]
        for index, row in count_objects.iterrows():
            if row[7] != 0:
                frame = display_objects(frame, row)
        
        if (TIME) >= display_time :
            FPS = fc / (TIME)
            fc = 0
            start_time = time.time()
        
        print("Frame number:"+ str(count)+"/"+str(total_frames)+" "+"FPS:"+str(FPS)[:5],end='\r')
        out.write(frame)
        if count == total_frames:
            break
    stream.release()

def test_live_video(name):
    '''A fucntion to test video annotations'''
    path1 = PATH + "videos_original\\"
    stream = cv2.VideoCapture(path1+name+".mp4")
    count = 0
    try:
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        print("error getting total frames")
        total_frames = "NaN"

    path = PATH + "annotations\\"
    fails = []
    objectonly = False
    try:
        events, objects, maps, assoc_maps = read_annots(path+name+".viratdata.events.txt", path+name+".viratdata.mapping.txt", path+name+".viratdata.objects.txt")
    except:
        try:
            objects = read_annots_object_only(path+name+".viratdata.objects.txt")
            objectonly = True
        except:
            fails.append(path+name+"does not exist, possible there are no actions in the video")
    if fails:
        print(fails)
        return
    start_time = time.time()
    # FPS update time in seconds
    display_time = 1
    fc=0
    FPS = 0
    while True:
        ret, frame = stream.read()
        count += 1
        fc+=1
        TIME = time.time() - start_time

        if not objectonly:
            count_events = events.loc[events[5] == count]
            for index, row in count_events.iterrows():
                if row[1] != 0:
                    frame = display_events(frame, row)

        count_objects = objects.loc[objects[2] == count]
        for index, row in count_objects.iterrows():
            if row[7] != 0:
                frame = display_objects(frame, row)

        if (TIME) >= display_time :
            FPS = fc / (TIME)
            fc = 0
            start_time = time.time()
        
        fps_disp = "FPS: "+str(FPS)[:5]
        frame = cv2.putText(frame, fps_disp, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.namedWindow("live_video", cv2.WINDOW_NORMAL)
        cv2.imshow("live_video", frame)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        print("Frame number:"+ str(count)+"/"+str(total_frames), end='\r')
        if  count == total_frames:
            break
    stream.release()
    cv2.destroyWindow("live_video")

def test_annots():
    '''Function to test all the annotations'''
    annots = PATH + "docs\\list_release2.0.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)

    path = PATH + "annotations\\"
    fails = []
    for i in range(len(annot_list)):
        #print(i)
        try:
            events, objects, maps, assoc_maps = read_annots(path+annot_list[0][i]+".viratdata.events.txt", path+annot_list[0][i]+".viratdata.mapping.txt", path+annot_list[0][i]+".viratdata.objects.txt")
        except:
            try:
                objects = read_annots_object_only(path+annot_list[0][i]+".viratdata.objects.txt")
            except:
                fails.append(str(annot_list[0][i])+"does not exist, possible there are no actions in the video")
    for i in fails:
        print(i)

def test_random_live_video():
    '''Function to test a random video from the dataset'''
    annots = PATH + "docs\\list_release2.0.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)
    rndint = random.randint(0, len(annot_list)-1)

    name = annot_list[0][rndint]
    test_live_video(name)

def test_random_video():
    '''Function to test a random video from the dataset'''
    annots = PATH + "docs\\list_release2.0.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)
    rndint = random.randint(0, len(annot_list)-1)

    name = annot_list[0][rndint]
    test_video(name)

def random_video():
    '''Function to test a random video from the dataset'''
    annots = PATH + "docs\\list_release2.0.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)
    rndint = random.randint(0, len(annot_list)-1)

    name = annot_list[0][rndint]
    return name


def extract_object_boxes(objects):
    '''Function to get all the annotation boxes
        returns the coords, class id and frame number'''
    x1 = objects[3]
    y1 = objects[4]
    x2 = objects[3]+objects[5]
    y2 = objects[4]+objects[6]

    coords = [x1, y1, x2, y2]
    class_id = objects[7]
    frame_no = objects[2]

    return coords, class_id, frame_no

def extract_event_boxes(events):
    '''Function to get all the annotation boxes
        returns the coords, class id and frame number'''
    x1 = events[6]
    y1 = events[7]
    x2 = events[6]+events[8]
    y2 = events[7]+events[9]

    coords = [x1, y1, x2, y2]
    class_id = events[1]
    frame_no = events[5]

    return coords, class_id, frame_no

def load_virat_dataset():
    annots = PATH + "docs\\list_release2.0.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)

    path = PATH + "annotations\\"
    fails = []
    for i in range(len(annot_list)):
        #print(i)
        try:
            events, objects, maps, assoc_maps = read_annots(path+annot_list[0][i]+".viratdata.events.txt", path+annot_list[0][i]+".viratdata.mapping.txt", path+annot_list[0][i]+".viratdata.objects.txt")
        except:
            try:
                objects = read_annots_object_only(path+annot_list[0][i]+".viratdata.objects.txt")
            except:
                fails.append(str(annot_list[0][i])+" does not exist")
        
        #event_coords, event_class_id, event_frame_no = extract_event_boxes(events)
        #obj_coords,   obj_class_id,   obj_frame_no   = extract_object_boxes(objects)
        #print(obj_class_id)
        #for j in range(len(objects)):
        #    if objects[7][j] == 0:
        #        print((annot_list[0][i]))
        #        break
        #path1 = PATH + "videos_original\\"
        #video = cv2.VideoCapture(path1+annot_list[0][i]+".mp4")
        #TODO: Next is to add these to a class or something#
        

    print(fails)
    #print(annot_list)

import sys
import torch
sys.path.append("...\\yolov7-main")
from utils.general import xyxy2xywh, xywh2xyxy

##Generate annotations for yolov
def generate_yolov_object_annotations():
    '''This function will generate annotations from the virat
    that are able to be used in the yolov object trainer/detector'''
    annots = PATH + "docs\\list_release2.0.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)

    print("Starting generation")
    path = PATH + "annotations\\"
    fails = []
    start_time = time.time()
    # FPS update time in seconds
    display_time = 1
    fc=0
    FPS = 0
    for i in range(len(annot_list)):
        save_dir = ('VIRATDATASET...VIRAT Ground Dataset\\Yolov_annotations\\')
        try:
            objects = read_annots_object_only(path+annot_list[0][i]+".viratdata.objects.txt")
        except:
            fails.append(str(annot_list[0][i])+" does not exist")
        Path(save_dir + str(annot_list[0][i])).mkdir(parents=True, exist_ok=True)
        save_dir = save_dir + str(annot_list[0][i])
        path1 = PATH + "videos_original\\"
        stream = cv2.VideoCapture(path1+str(annot_list[0][i])+".mp4")
        ret, frame = stream.read()
        counter = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        stream.release()
        for count in range(counter):
            fc+=1
            TIME = time.time() - start_time
            if (TIME) >= display_time :
                FPS = fc / (TIME)
                fc = 0
                start_time = time.time()
            print("File: " + str(i) + "/" + str(len(annot_list)) + " Running at: " + str(FPS)[:5] + " fps", end='\r')
            count_objects = objects.loc[objects[2] == count]
            for index, row in count_objects.iterrows(): 
                x1 = row[3]
                y1 = row[4]
                x2 = row[3]+row[5]
                y2 = row[4]+row[6]
                xyxy= [x1,y1,x2,y2]
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (row[7], *xywh)
                with open(str(save_dir) + "\\" + str(annot_list[0][i]) + "_" + str(count) + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
    print(fails)

def display_yolov_objects(frame, objects):
    '''Function to display all the objects in a video'''
    colour = object_dict[object_types[int(objects[0])]]
    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
    xywh = [objects[1], objects[2], objects[3], objects[4]]
    xyxy = (xywh2xyxy(torch.tensor(xywh).view(1, 4)) * gn).view(-1).tolist()
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])

    image = cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    image = cv2.putText(
        image, object_types[int(objects[0])], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, colour, 2
        )
    return image

def test_live_video_yolov_annotation(name):
    '''A fucntion to test video annotations'''
    path1 = PATH + "videos_original\\"
    stream = cv2.VideoCapture(path1+name+".mp4")
    count = 0
    try:
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        print("error getting total frames")
        total_frames = "NaN"

    path = PATH + "Yolov_annotations\\" + name + "\\"
    fails = []
    start_time = time.time()
    # FPS update time in seconds
    display_time = 1
    fc=0
    FPS = 0
    while True:
        ret, frame = stream.read()
        fc+=1
        TIME = time.time() - start_time

        try:
            objects = read_annots_object_only(path + name  + "_" + str(count) + ".txt")
            for index, row in objects.iterrows():
                if row[0] != 0:
                    frame = display_yolov_objects(frame, row)
        except:
            fails.append(name + " empty frame number " + str(count))
    
        if (TIME) >= display_time:
            FPS = fc / (TIME)
            fc = 0
            start_time = time.time()
        
        fps_disp = "FPS: "+str(FPS)[:5]
        frame = cv2.putText(frame, fps_disp, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.namedWindow("live_video", cv2.WINDOW_NORMAL)
        cv2.imshow("live_video", frame)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        print("Frame number:"+ str(count)+"/"+str(total_frames), end='\r')
        count += 1
        if  count == total_frames:
            break
    print("\n")
    print(fails)
    stream.release()
    cv2.destroyWindow("live_video")

def split_video_yolov(name):
    path1 = PATH + "videos_original\\"
    stream = cv2.VideoCapture(path1+name+".mp4")
    count = 0
    try:
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        print("error getting total frames")
        total_frames = "NaN"

    path = "VIRATDATASET...\\VIRAT Ground Dataset\\Yolov_images\\"+name+"\\"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start_time = time.time()
    # FPS update time in seconds
    display_time = 1
    fc=0
    FPS = 0
    while True:
        ret, frame = stream.read()
        fc+=1
        TIME = time.time() - start_time

        try:
            cv2.imwrite(path+name+"_"+str(count)+".jpg", frame)
        except:
            print("Frame " + str(count) + " failed from video " + str(name))

        if (TIME) >= display_time:
            FPS = fc / (TIME)
            fc = 0
            start_time = time.time()
        
        fps_disp = " FPS: "+str(FPS)[:5]
        print("Frame number:"+ str(count)+"/"+str(total_frames) + fps_disp, end='\r')
        count += 1
        if  count == 500:
            break
    stream.release()

def split_all_yolov_videos():
    '''Function to split all of the videos so they can be trained on yoloc'''
    annots = PATH + "docs\\list_release2.0.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)

    print("Starting generation")
    path = PATH + "videos_original\\"
    for i in range(len(annot_list)):
        print("\n"+"Done " + str(i) + " vidoes\n")
        split_video_yolov(str(annot_list[0][i]))
    
##Generate annotations for yolov
def generate_yolov_action_annotations():
    '''This function will generate annotations from the virat
    that are able to be used in the yolov object trainer/detector'''
    annots = PATH + "docs\\list_release2.0edit.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)

    print("Starting generation")
    path = PATH + "annotations\\"
    fails = []
    start_time = time.time()
    # FPS update time in seconds
    display_time = 1
    fc=0
    FPS = 0
    for i in range(len(annot_list)):
        save_dir = ('VIRATDATASET...\\VIRAT Ground Dataset\\action_annotations\\')
        try:
            events = read_annots_events_only(path+annot_list[0][i]+".viratdata.events.txt")
        except:
            fails.append(str(annot_list[0][i])+" does not exist")
        Path(save_dir + str(annot_list[0][i])).mkdir(parents=True, exist_ok=True)
        save_dir = save_dir + str(annot_list[0][i])
        path1 = PATH + "videos_original\\"
        stream = cv2.VideoCapture(path1+str(annot_list[0][i])+".mp4")
        ret, frame = stream.read()
        counter = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        stream.release()
        for count in range(counter):
            fc+=1
            TIME = time.time() - start_time
            if (TIME) >= display_time :
                FPS = fc / (TIME)
                fc = 0
                start_time = time.time()
            print("File: " + str(i) + "/" + str(len(annot_list)) + " Running at: " + str(FPS)[:5] + " fps", end='\r')
            count_events = events.loc[events[5] == count]
            for index, row in count_events.iterrows(): 
                x1 = row[6]
                y1 = row[7]
                x2 = row[6]+row[8]
                y2 = row[7]+row[9]
                xyxy = [x1,y1,x2,y2]
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (row[1], *xywh)
                with open(str(save_dir) + "\\" + str(annot_list[0][i]) + "_" + str(count) + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
    print(fails)

def split_action_video_yolov(name):
    path1 = PATH + "videos_original\\"
    stream = cv2.VideoCapture(path1+name+".mp4")
    count = 0
    try:
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        print("error getting total frames")
        total_frames = "NaN"

    path = "VIRATDATASET...\\VIRAT Ground Dataset\\action_images\\"+name+"\\"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start_time = time.time()
    # FPS update time in seconds
    text = "VIRATDATASET...\\VIRAT Ground Dataset\\action_annotations\\"+name+"\\"
    display_time = 1
    fc=0
    FPS = 0
    while True:
        ret, frame = stream.read()
        fc+=1
        TIME = time.time() - start_time

        textpath = text + name + "_" + str(count) + ".txt"
        if (os.path.isfile(textpath)):
            try:
                cv2.imwrite(path+name+"_"+str(count)+".jpg", frame)
            except:
                print("Frame " + str(count) + " failed from video " + str(name))

        if (TIME) >= display_time:
            FPS = fc / (TIME)
            fc = 0
            start_time = time.time()
        
        fps_disp = " FPS: "+str(FPS)[:5]
        print("Frame number:"+ str(count)+"/"+str(total_frames) + fps_disp, end='\r')
        if count == total_frames:
            break
        count += 1
    stream.release()

def split_all_yolov_action_videos():
    '''Function to split all of the videos so they can be trained on yoloc'''
    annots = PATH + "docs\\list_release2.0.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)

    print("Starting generation")
    path = PATH + "videos_original\\"
    for i in range(len(annot_list)):
        print("\n"+"Done " + str(i) + " vidoes\n")
        split_action_video_yolov(str(annot_list[0][i]))

def create_tublet_videos():
    '''function to split the action videos into tublets with classes'''
    annots = PATH + "docs\\list_release2.0.txt"
    annot_list = pd.read_csv(annots, sep=" ", header=None)
    annots_path = PATH + "annotations\\"
    videos_path = PATH + "videos_original\\"
    for i in range(len(annot_list)):
        print("\n"+"Done " + str(i) + " vidoes and annotations\n")
        name = str(annot_list[0][i])
        path = PATH + "annotated_tublets\\videos\\"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = PATH + "annotated_tublets\\class_label\\"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = PATH + "annotated_tublets\\"
        try:
            video_annotations = read_annots_events_only(annots_path+name+".viratdata.events.txt")
        except:
            print("cannot read annotations: " + name)
        if len(video_annotations)>0:
            stream = cv2.VideoCapture(videos_path+name+".mp4")
            create_tublets(stream, video_annotations, name, path)
        video_annotations = []
        stream.release()

        
def create_tublets(stream, annotations, name, save_path):
    '''function to split a video into action tublets and save them'''
    #give each tublet a unique id
    idnum = 0
    #split into videos and class_label
    class_path = save_path + "class_label\\"
    video_path = save_path + "videos\\"
    outs = []
    ids = []
    wh = []
    maxwidth = 0
    maxheight = 0
    old_row = annotations[0][0]
    old_class = 0
    for index, row in annotations.iterrows(): 
        if row[0] > old_row:
            outs.append(cv2.VideoWriter(video_path+name+'_'+str(old_row)+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(maxwidth),int(maxheight))))
            #append to dict
            ids.append(old_row)
            wh.append([maxwidth, maxheight])
            line = str(old_class)
            with open(class_path + name + "_" + str(old_row) + '.txt', 'w') as f:
                f.write(line)
            maxwidth = 0
            maxheight = 0
            idnum+=1
        #classid = row[1]
        if row[8] > maxwidth:
            maxwidth = row[8]
        if row[9] > maxheight:
            maxheight = row[9]
        #width = row[8]
        #height = row[9]
        if index == annotations.shape[0]-1:
            outs.append(cv2.VideoWriter(video_path+name+'_'+str(row[0])+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(maxwidth),int(maxheight))))
            ids.append(row[0])
            wh.append([maxwidth, maxheight])
            line = str(row[1])
            with open(class_path + name + "_" + str(row[0]) + '.txt', 'w') as f:
                f.write(line)
        old_row = row[0]
        old_class = row[1]
    try:
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        print("error getting total frames")
        total_frames = "NaN"
    count = 0
    start_time = time.time()
    display_time = 1
    fc=0
    FPS = 0
    while True:
        ret, frame = stream.read()
        fc+=1
        TIME = time.time() - start_time

        #split the video according to the unique event ids
        df2 = annotations.loc[annotations[5] == count]

        for index, row in df2.iterrows():
            bboxframe = frame[row[7]:row[7]+row[9] , row[6]:row[6]+row[8]]
            indexof = ids.index(row[0])
            bboxframe = cv2.resize(bboxframe,(int(wh[indexof][0]), int(wh[indexof][1])),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            outs[indexof].write(bboxframe)

        if (TIME) >= display_time:
            FPS = fc / (TIME)
            fc = 0
            start_time = time.time()
        
        fps_disp = " FPS: "+str(FPS)[:5]
        print("Frame number:"+ str(count)+"/"+str(total_frames) + fps_disp, end='\r')
        if count == total_frames-1:
            break
        count += 1
    for out in outs:
        out.release()

import shutil

def split_test_train_val():
    '''function to split the action tubes into test train val
    datasets'''
    file_path = PATH + "annotated_tublets\\"
    videos_path = file_path + "videos\\"
    class_path = file_path + "class_label\\"

    train_save_path = "...\\action_detection\\data\\train"
    test_save_path  = "...\\action_detection\\data\\test"
    val_save_path   = "...\\action_detection\\data\\val"

    count = 0
    #split: 80, 10, 10
    for filename in os.listdir(videos_path):
        v = os.path.join(videos_path, filename)
        # checking if it is a file
        if os.path.isfile(v):
            name = (filename[:-4])
            rnd = random.randint(0, 100)
            if (rnd <= 80):
                shutil.copyfile(v, train_save_path+"\\videos\\"+name+".mp4")
                shutil.copyfile(class_path+name+".txt", train_save_path+"\\class_label\\"+name+".txt")
            elif (rnd > 80) and (rnd < 90):
                shutil.copyfile(v, test_save_path+"\\videos\\"+name+".mp4")
                shutil.copyfile(class_path+name+".txt", test_save_path+"\\class_label\\"+name+".txt")
            else:
                shutil.copyfile(v, val_save_path+"\\videos\\"+name+".mp4")
                shutil.copyfile(class_path+name+".txt", val_save_path+"\\class_label\\"+name+".txt")
        count+=1
        print("Done "+ str(count) +" video annotation file", end='\r')
    print("\n Done")
