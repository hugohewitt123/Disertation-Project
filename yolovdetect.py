import time
from pathlib import Path

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import cv2
import math
import torch
from random import randint
from operator import itemgetter
import torch.backends.cudnn as cudnn
from numpy import random
from inference_action_model import (initialize_action_model, detect_action)

import sys
sys.path.append("C:\\Users\\hugo\\Documents\\University\\YEAR_3\\ECM3401_Project\\Action Detection\\yolov7-main")

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, xywh2xyxy
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from sort import *

def yolov_object_detect(source, weights, save_img, view_img, save_txt=False, imgsz=640, trace=False):
    '''Function to do object detection on a video'''
    # Directories
    save_dir = Path(increment_path(Path('runs/detect') / 'exp', exist_ok=True))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    #Device
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size


    if trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    start_time = time.time()
    # FPS update time in seconds
    display_time = 1
    fc=0
    FPS = 0
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        fgmask = fgbg.apply(im0s)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        fc+=1
        TIME = time.time() - start_time

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=[0, 1, 2, 3, 5, 7], agnostic=False)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if True else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            ### See utils/datasets.py line 181/182 for full print sequence ###

            #print(frame,end='\r')

            if (TIME) >= display_time :
                FPS = fc / (TIME)
                fc = 0
                start_time = time.time()

            # Stream results
            if view_img:
                fps_disp = "FPS: "+str(FPS)[:5]
                im0 = cv2.putText(im0, fps_disp, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.namedWindow("videostream", cv2.WINDOW_NORMAL)
                cv2.imshow("videostream", im0)
                cv2.namedWindow("segement", cv2.WINDOW_NORMAL)
                cv2.imshow("segement", fgmask)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    print("\n")
                    cv2.destroyAllWindows()
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

def draw_roi_boxes(img, colors, bbox, ACTION_NAMES, categories=None, identities=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):
    """Function to Draw Bounding boxes from the object tracker"""
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = str(round(identities[i],3)) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(ACTION_NAMES[cat] + " score " + id)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[int(cat)], 2)
        cv2.rectangle(img, (x1, y1 - 30), (x1 + int(w*1.2), y1), colors[int(cat)], -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, [255,255,255], 2)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0])/img.shape[1], int(box[1])/img.shape[0] , int(box[2])/img.shape[1], int(box[3])/img.shape[0] ,int(box[0] + (box[2] * 0.5))/img.shape[1] ,
                int(box[1] + (
                    box[3]* 0.5))/img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img

def draw_boxes(img, colors, bbox, conf, identities=None, categories=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):
    """Function to Draw Bounding boxes from the object tracker"""
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        confd = int(round(conf[i]*100)) if conf is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        if conf is not None:
            label = str(id) + ":"+ names[cat] + " " + str(confd) + "%"
        else:
            label = "Proposal"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[int(cat)], 2)
        cv2.rectangle(img, (x1, y1 - 30), (x1 + int(w*1.2), y1), colors[int(cat)], -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, [255,255,255], 2)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0])/img.shape[1], int(box[1])/img.shape[0] , int(box[2])/img.shape[1], int(box[3])/img.shape[0] ,int(box[0] + (box[2] * 0.5))/img.shape[1] ,
                int(box[1] + (
                    box[3]* 0.5))/img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img

def generate_proposals(tracker_dets, tracks, names):
    '''Function to generate action proposals from 
    the obejct detection and tracking information '''
    rois = []
    if len(tracker_dets)>0:
        bbox_xyxy = tracker_dets[:,:4].tolist()
        identities = tracker_dets[:, 8]
        categories = tracker_dets[:, 4]
    trks = np.zeros((len(tracks), 6))
    for t, trk in enumerate(trks):
        pos = tracks[t].bbox_history[-1]
        trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
        if np.any(np.isnan(pos)):
            to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

    matched, unmatched_dets, unmatched_tracks = associate_detections_to_trackers(tracker_dets, trks)

    ids = []
    for i, m in enumerate(matched):
        bbox_history = []
        hbox = tracker_dets[m[0]][:4].tolist()
        hbox = (xyxy2xywh(torch.tensor(hbox).view(1, 4))).view(-1).tolist()
        w = hbox[2] + (hbox[2]*1.2)
        h = hbox[3] + (hbox[3]/2)
        new_box = [hbox[0], hbox[1], w, h]
        abox = (xywh2xyxy(torch.tensor(new_box).view(1, 4))).view(-1).tolist()

        id1 = (tracker_dets[m[0]][8])

        #this only adds person objects to the proposals
        if tracker_dets[m[0]][4] == 0:
            rois.append(abox)
            ids.append(id1)
    return rois, ids

def generate_predict_proposals(tracker_dets, tracks, names):
    '''Function to generate action proposals from 
    the obejct detection and tracking information '''
    rois = []
    if len(tracker_dets)>0:
        bbox_xyxy  = tracker_dets[:, 0]
        identities = tracker_dets[:, 1]
        categories = tracker_dets[:, 3]
    else:
        return
    trks = np.zeros((len(tracks), 6))
    for t, trk in enumerate(trks):
        pos = tracks[t].bbox_history[-1]
        trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
        if np.any(np.isnan(pos)):
            to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

    ids = []
    for i, hbox in enumerate(bbox_xyxy):
        #hbox = tracker_dets[m[0]][:4].tolist()
        hbox = (xyxy2xywh(torch.tensor(hbox).view(1, 4))).view(-1).tolist()
        w = hbox[2] + (hbox[2]*1.2)
        h = hbox[3] + (hbox[3]/2)
        new_box = [hbox[0], hbox[1], w, h]
        abox = (xywh2xyxy(torch.tensor(new_box).view(1, 4))).view(-1).tolist()

        id1 = (identities[i])

        #only add person to the proposals
        if categories[i] == 0:
            rois.append(abox)
            ids.append(id1)
    #print("\n")
    return rois, ids

def refine_id_proposals(rois, fgmask, ids, threshold=4):
    '''function to filter out the poor proposals'''
    refined_rois = []
    refined_ids = []
    mean = 0
    for i, roi in enumerate(rois):
        box = (xyxy2xywh(torch.tensor(roi).view(1, 4))).view(-1).tolist()
        mask = fgmask[int(box[1]-(box[3]/1.5)):int(box[1]+(box[3]/1.5)), int(box[0]-(box[2]/2)):int(box[0]+(box[2]/2))]
        #applying the binary mask averages and filtering
        #out those with scores below the threshold
        if mask.any():
            mean = mask.mean()
        if mean > threshold:
            refined_rois.append(roi)
            refined_ids.append(ids[i])
    return refined_rois, refined_ids


### Object detection,tracking and action detection ###
def yolov_object_n_action_detect(source, weights, save_img, view_img, save_txt=False, imgsz=640, trace=False, show_tracking=False, skip_rate=1, draw_action_boxes=True, draw_object_boxes=False, draw_proposal_boxes=False, show_foregroundseg=False, threshold=5):
    '''Function to perform action detection on a video'''
    save_bbox_dim = False
    save_with_object_id = False
    
    #SORT tracking parameters
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
 
    # Directories
    action_proposal_frames = []
    ID = 0
    
    save_dir = Path(increment_path(Path('runs/detect') / 'exp', exist_ok=False))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    #Device
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    action_model = initialize_action_model()

    if trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    ACTION_NAMES =  ["na", 
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
    actioncolors = [[random.randint(0, 255) for _ in range(3)] for _ in ACTION_NAMES]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    start_time = time.time()
    reset_time = time.time()
    # FPS update time in seconds
    display_time = 1
    fc = 0
    FPS = 30
    count = 0
    trackers = None
    last10frames = []
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    for path, img, im0s, vid_cap in dataset:
        last10frames.append(im0s)
        if len(last10frames) > 10:
            last10frames.pop(0)
        #Background removal with foreground segmentation
        fgmask = fgbg.apply(im0s)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        fc+=1
        TIME = time.time() - start_time
        trks = []
        trackers = []

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        if (count%skip_rate == 0):
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred_new = non_max_suppression(pred, 0.25, 0.45, classes=[0, 1, 2, 3, 5, 7], agnostic=False)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred_new = apply_classifier(pred_new, modelc, img, im0s)
        else:
            pred_new = []
            im0 = im0s
            tracks = sort_tracker.getTrackers()
            for i, track in enumerate(tracks):
                temp = tracks[i].predict_object().tolist()
                temp.append(track.id+1)
                temp.append(track.conf)
                temp.append(track.bbox_history[-1][5])
                trks.append(temp)
            if len(trks)>0 and draw_object_boxes:
                bbox_xyxy = list(map(itemgetter(0), trks))
                identities = list(map(itemgetter(1), trks))
                conf = list(map(itemgetter(2), trks))
                categories = list(map(itemgetter(3), trks))
                im0 = draw_boxes(im0, colors, bbox_xyxy, conf, identities, categories, names, save_with_object_id, txt_path)

        # Process detections
        for i, det in enumerate(pred_new):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                #### tracking the objects with sort ####
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets, trackers = sort_tracker.update(dets_to_sort)
                
                if show_tracking:
                    tracks = sort_tracker.getTrackers()

                    txt_str = ""
                    #loop over tracks
                    for track in tracks:
                        # color = compute_color_for_labels(id)
                        #draw colored tracks
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    (255,0,0), thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                        if i < len(track.centroidarr)-1 ] 

                # draw boxes for visualization
                if len(tracked_dets)>0 and draw_object_boxes:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    conf = tracked_dets[:, 9]
                    categories = tracked_dets[:, 4]
                    im0 = draw_boxes(im0, colors, bbox_xyxy, conf, identities, categories, names, save_with_object_id, txt_path)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            ### See utils/datasets.py line 181/182 for full print sequence ###

            #print(frame,end='\r')

        ##TODO make this more efficient
        if trackers:
            rois, ids = generate_proposals(tracked_dets, trackers, names)
            #TODO next is implement frame skipping
        elif trks:
            rois, ids = generate_predict_proposals(np.array(trks, dtype=object), tracks, names)
        if trackers or trks:
            rois, ids = refine_id_proposals(rois, fgmask, ids, threshold)
            if draw_proposal_boxes:
                im0 = draw_boxes(im0, actioncolors, rois, None)
            for i, roi in enumerate(rois):
                xywh = (xyxy2xywh(torch.tensor(roi).view(1, 4))).view(-1).tolist()
                w = xywh[2]
                h = xywh[3]
                nfound = True
                for actionfprop in action_proposal_frames:
                    if int(actionfprop[0][0]) == ids[i]:
                        actionfprop.append([ids[i], roi, int(roi[1]), int(roi[1]+h) , int(roi[0]), int(roi[0]+w)])
                        nfound = False
                    if len(actionfprop) > 10:
                        actionfprop.pop(0)
                if nfound:
                    action_proposal_frames.append([[ids[i], roi, int(roi[1]), int(roi[1]+h) , int(roi[0]), int(roi[0]+w)]])
            for i, actionpropthiing in enumerate(action_proposal_frames):
                if actionpropthiing[0][0] in ids:
                    None
                else:
                    action_proposal_frames.pop(i)

            class_ids = []
            actions = []
            scores = []
            i=0
            for i, tubelet in enumerate(action_proposal_frames):
                if len(tubelet) == 10:
                    try:
                        classid, score = detect_action(action_model, tubelet, last10frames)
                        if classid:
                            class_ids.append(classid)
                            scores.append(score)
                            actions.append(tubelet[9][1])
                    except:
                        print("one failed action detect")
            if draw_action_boxes:
                im0 = draw_roi_boxes(im0, actioncolors, actions, ACTION_NAMES, class_ids, scores)

        
        if (TIME) >= display_time:
            FPS = fc / (TIME)
            fc = 0
            start_time = time.time()

        # Stream results
        if view_img:
            fps_disp = "FPS: " + str(FPS)[:5]
            im0 = cv2.putText(im0, fps_disp, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.namedWindow("videostream", cv2.WINDOW_NORMAL)
            cv2.imshow("videostream", im0)
            #Used to show the foreground segmentation
            if show_foregroundseg:
                cv2.namedWindow("segement", cv2.WINDOW_NORMAL)
                cv2.imshow("segement", fgmask)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                print("\n")
                cv2.destroyAllWindows()
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
        count+=1

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    print("\n")
    print(f'Done. ({time.time() - t0:.3f}s)')