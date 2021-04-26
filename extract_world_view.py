import os
import sys
import cv2
import time
import json
import random
import numpy as np
import pandas as pd
import torch, torchvision

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def predict(im):
    outputs = predictor(im)
    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    #print(outputs["instances"].pred_classes)
    #print(outputs["instances"].pred_boxes)
    return outputs

def visualize(im, outputs, gaze_pos):
    
    output_instances = outputs["instances"].to("cpu")

    ''' v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(output_instances)
    out = out.get_image()[:, :, ::-1] '''
    
    humans = output_instances.pred_classes == 0
    besthuman_index = humans.nonzero()[0].item()
    besthuman_mask = output_instances.pred_masks[besthuman_index].numpy()
    besthuman_box = output_instances.pred_boxes[besthuman_index].tensor[0].numpy().astype(int)
    start_x, start_y, end_x, end_y = besthuman_box

    # Check if given pixel location of gaze tracking is labeled as baby
    # 1. Based on segmentation 
    print("1. Segmentation: Looking to the baby??", besthuman_mask[gaze_pos[0], gaze_pos[1]])
    # 2. Based on bounding box 
    print("2. Bounding box: Looking to the baby??", (start_x <= gaze_pos[0] <= end_x) and (start_y <= gaze_pos[1] <= end_y))

    out = cv2.circle(im, gaze_pos, 3, (0,0,255), -1)
    out = cv2.rectangle(out, (start_x, start_y), (end_x, end_y),  (255, 0, 0), 2)
    cv2.cv2_imshow(out)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

if __name__  == "__main__":

    # Get files
    data_dir = sys.argv[-1].split('/')
    session_number = data_dir[-1]
    try: 
        date_recording = data_dir[-2]
    except IndexError:
        date_recording = ''

    recording_dir = os.path.join(date_recording, session_number)
    gaze_data_dir = os.path.join(recording_dir, 'exports', os.listdir(os.path.join(recording_dir, 'exports'))[0])
    gaze_df = pd.read_csv(os.path.join(gaze_data_dir, 'gaze_positions.csv'))
    
    # Get gaze data
    world_vid = os.path.join(recording_dir, 'world.mp4')
    cap = cv2.VideoCapture(world_vid)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 1280 px
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 720 px
    print("frame rate:", frame_rate, "width:", width, "height:", height)

    # Export
    os.makedirs(os.path.join("output", date_recording), exist_ok=True)
    video_out = cv2.VideoWriter(filename=os.path.join("output", date_recording, f"{session_number}_world_view_with_detection.avi"), 
                                apiPreference=cv2.CAP_ANY,
                                fourcc=cv2.VideoWriter_fourcc(*"XVID"), 
                                fps=frame_rate, 
                                frameSize=(width, height))

    # Create a detectron2 config using different configurations:
    # Find a model from detectron2's model zoo: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    #   COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml -  44.3	39.5
    #   Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml - 50.2  44.0
    #   PascalVOC-Detection/faster_rcnn_R_50_C4.yaml - bad, many false positive 
    #   COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
    #   COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))     
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in range(frame_count):
        ret, frame = cap.read()
        if ret:
            gaze_datum = gaze_df[gaze_df['world_index']==f]
            x_norm_coor, y_norm_coor = gaze_datum['norm_pos_x'].to_numpy(), gaze_datum['norm_pos_y'].to_numpy()
            #print(x_norm_coor, y_norm_coor) 

            x_img_coor, y_img_coor = (x_norm_coor*width).astype(int), ((1-y_norm_coor)*height).astype(int)
            for sub_frame in range(len(gaze_datum)):
                # Predict
                prediction = predict(frame)

                # Visualize
                gaze = (x_img_coor[sub_frame], y_img_coor[sub_frame])
                #print(gaze)
                pred = visualize(frame, prediction, gaze)
            time.sleep(0.01) # Add some delay to avoid processing too fast
            
            #cv2.imshow('Recording', frame)
            #video_out.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()

    cap.release()
    video_out.release() 

# EOF