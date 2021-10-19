import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog

def get_gaze_thres(gaze_df, frame_count):
    """Get the lowest threshold of confidence user can set so that there is at
    least one gaze point in each frame
    
    :param gaze_df: dataframe containing gaze info
    :param frame_count: number of video's frames

    :return threshold: lowest threshold
    :rtype: float
    """

    threshold_list = gaze_df['confidence'].drop_duplicates().sort_values()
    for threshold in threshold_list:
        good_gaze_df = gaze_df.loc[gaze_df['confidence'] > threshold] 
        if good_gaze_df['world_index'].nunique() < frame_count:
            return threshold

def check_gaze_in_detection(gaze_pos, mask, box, frame):
    """Check if gaze point (calculated) is within segmentation and bounding box
    or not and assign gaze's color in each case (green - face is detected and 
    gaze is in face's bounding box, red - face is detected but gaze is not in 
    face's box, yellow - face is not detected)

    :param gaze_pos: coordinate of gaze
    :param mask: bitmask for the detected baby, same shape with video frame 
    :param box: coordinates of upper left and bottom right corners of the box
    :param frame: video frame

    :return in_segment: whether gaze is in segmentation or not
    :rtype: boolean
    :return in_box: whether gaze is in bounding box or not
    :rtype: boolean
    :return gaze_color: RGB value for the gaze
    :rtype: tuple
    """
    
    gaze_color_dict = {
        "gaze_in_detection" : (0, 255, 0),
        "gaze_not_in_detection" : (0, 0, 255),
        "not_detected" : (0, 255, 255)
    }

    if (mask is None) and (box is None):
        return np.nan, np.nan, gaze_color_dict["not_detected"]

    # Clipping values if gaze is outside the world view
    gaze_x, gaze_y = np.clip(gaze_pos, 0, (mask.shape[1] - 1, mask.shape[0] - 1))

    # In on segmentation?
    in_segment = mask[gaze_y, gaze_x]

    # In bounding box?
    (start_x, start_y), (end_x, end_y) = np.floor(box[:2]), np.ceil(box[2:])
    in_box = (start_x <= gaze_x <= end_x) and (start_y <= gaze_y <= end_y)

    # Assign color to gaze
    gaze_color = gaze_color_dict["gaze_in_detection"] if in_box else gaze_color_dict["gaze_not_in_detection"]

    return in_segment, in_box, gaze_color
    
def get_config(config_file):
    """Get a predictor and video visualizer given the configuration file

    :param config_file: Configuration file
    :param config_file: string

    :return predictor: End-to-end predictor object with the given config that
    runs on single device for a single input image
    :rtype: detectron2.engine.defaults.DefaultPredictor
    :return visualizer: Visualizer object that draws data about detection,
    segmentation on video's frames 
    :rtype: detectron2.utils.video_visualizer.VideoVisualizer
    """

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))     
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model, faster inference
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    if not torch.cuda.is_available():
        print("CUDA is not available", end='\n\n')
        cfg.MODEL.DEVICE = "cpu"
    print(f"Run on CUDA: {torch.cuda.get_device_name(0)}", end='\n\n')
    return DefaultPredictor(cfg), VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

def detect_person(im, predictor):
    """Find persons in the image

    :param im: Image in the format of BGR with shape (H, W, 3)
    :type im: numpy.ndarray
    :param predictor: DefaultPredictor object 
    :type predictor: predictor object

    :return baby_instance: Instance object containing prediction's attributes
    of all person
    :rtype: detectron2.structures.Instances
    """

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    outputs = predictor(im)
    output_instances = outputs["instances"].to("cpu")
    person_id = predictor.metadata.thing_classes.index('person')
    person_instances = output_instances[output_instances.pred_classes == person_id]
    return person_instances

def visualize(frame, pred_instance, visualizer, show=False, rgb=False):
    """Visualize result from predictor

    :param frame: BGR video frame with shape (H, W, 3)
    :type frame: numpy.ndarray
    :param pred_instance: Instance object of a prediction result from predictor
    :type pred_instance: detectron2.structures.Instances
    :param visualizer: Visualizer objects
    :type visualizer: detectron2.utils.video_visualizer.VideoVisualizer

    :return: Segmentation, bounding box of the baby, and the gaze point in the
    frame (BGR format as default)
    :rtype: numpy.ndarray
    """
    
    out = visualizer.draw_instance_predictions(frame[:, :, ::-1], pred_instance)
    out = out.get_image()   # RGB format
    if show: 
        cv2.imshow(out)
    return out if rgb else cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def get_gaze_in_frame(gaze_series, width, height):
    """Calculate location of gaze corresponding to the video frame coordinate

    :param gaze_series: all gazes in the video frame
    :param width: width of the video frame
    :param height: height of the video frame

    :return x_img_coor: x coordinate of the gaze
    :rtype: int
    :return y_img_coor: y coordinate of the gaze
    :rtype: int
    """

    x_norm_coor, y_norm_coor = gaze_series['norm_pos_x'].to_numpy(), gaze_series['norm_pos_y'].to_numpy()
    x_img_coor, y_img_coor = (x_norm_coor*width).astype(int), ((1-y_norm_coor)*height).astype(int)
    return x_img_coor, y_img_coor

def baby_detection(recording_dir, predictor, visualizer, gaze_thres=0.85):
    """Find baby in the video and check if gaze is at the baby or not

    :param recording_dir: directory of gaze
    :param predictor: predictor used to find baby
    :param visualizer: VideoVisualizer object
    :param gaze_thres: lowest confidence level of each gaze set by user
    """

    # run predictor on each image, if no baby is found, then output image with gaze point only
    # if there's a baby, then output image with gaze point, segmentation, (bounding box if possible)
    # Write to the table whether that gaze in the box? in the segmentation?
    
    # Get gaze data
    gaze_data_dir = os.path.join(recording_dir, 'exports', os.listdir(os.path.join(recording_dir, 'exports'))[0])
    raw_gaze_df = pd.read_csv(os.path.join(gaze_data_dir, 'gaze_positions.csv'), usecols=["world_index", "confidence", "norm_pos_x", "norm_pos_y"])

    # Get video
    world_vid = os.path.join(recording_dir, 'world.mp4')
    video_cap = cv2.VideoCapture(world_vid)
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 1280 px
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 720 px
    print("Frame rate:", frame_rate, "number of frames", frame_count, "width:", width, "height:", height)

    # Filter low confidence gaze base on user's choice of gaze_thres
    min_thres = get_gaze_thres(raw_gaze_df, frame_count)
    if gaze_thres >= min_thres:
        min_thres = min_thres // 0.01 * 0.01    # Truncate number to 2 decimal places
        sys.exit(f"Should use threshold smaller than or equal to {min_thres} to avoid losing frames") 
    gaze_df = raw_gaze_df.loc[raw_gaze_df['confidence'] > gaze_thres].copy()
    discarded = 1 - len(gaze_df)/len(raw_gaze_df)
    print(f"{discarded*100:.2f}% of gaze points is discarded due to low confidence (<{gaze_thres})")
    
    # Prepare output video
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")   # Using current date and time for unique file name
    video_out = cv2.VideoWriter(filename=os.path.join(recording_dir, f"world_view_with_detection_{dt_string}.avi"), 
                                apiPreference=cv2.CAP_ANY,
                                fourcc=cv2.VideoWriter_fourcc(*"XVID"), 
                                fps=frame_rate, 
                                frameSize=(width, height))
    
    # Start processing
    start = time.time()
    detect_baby = []
    gaze_in_segment = []
    gaze_in_box = []
    for frame_ind in range(frame_count):
        retval, frame = video_cap.read()
        if retval:
            # Get all gazes data in the frame
            gaze_datum = gaze_df.loc[gaze_df['world_index'] == frame_ind]
            num_gaze = len(gaze_datum)

            # Initialize key components
            segmentation = None
            bounding_box = None
            vis_frame = frame

            # Detectron work
            detection = detect_person(frame, predictor)   
            if len(detection) == 0:  # If no human presents in the frame
                detect_baby.extend([False]*num_gaze)
            else:
                detect_baby.extend([True]*num_gaze)
                # Only get the most confidence person instance (assume the baby is the only person in the video)
                baby_instance = detection[0]  
                segmentation = baby_instance.pred_masks.numpy().squeeze()
                bounding_box = baby_instance.pred_boxes.tensor.numpy().squeeze()
                
                # Visualization
                vis_frame = visualize(frame, baby_instance, visualizer)    

            # Check gaze in detection
            gaze_list_x, gaze_list_y = get_gaze_in_frame(gaze_datum, width, height)
            for gaze_ind in range(num_gaze):
                gaze_pos = (gaze_list_x[gaze_ind], gaze_list_y[gaze_ind])
                in_segmentation, in_box, gaze_color = check_gaze_in_detection(gaze_pos, segmentation, bounding_box, f"Frame: {frame_ind}.{gaze_ind}")
                gaze_in_segment.append(in_segmentation)
                gaze_in_box.append(in_box)

                # Draw on visualization
                # Have to draw on vis_frame.copy() if one wants to add some transparent to gaze point
                vis_frame = cv2.circle(img=vis_frame, 
                                       center=gaze_pos, 
                                       radius=4, 
                                       color=gaze_color,
                                       thickness=-1)  
                
                #print("Segmentation: Looking to the baby?", in_segmentation)
                #print("Bounding box: Looking to the baby?", in_box)
                #print()
            
            # Write to output video
            video_out.write(vis_frame)
            #time.sleep(0.01) # Add some delay to see processing frames
            #cv2.imshow(window_name='Recording', 
            #            image=vis_frame)
            
            
            if cv2.waitKey(delay=1) & 0xFF == ord('q'):
                sys.exit()
    
    video_cap.release()
    video_out.release() 
    print("Total run time", time.time() - start)
    
    # Write result to csv file 
    gaze_df["is_baby"] = detect_baby
    gaze_df["in_segmentation"] = gaze_in_segment
    gaze_df["in_bounding_box"] = gaze_in_box
    gaze_df.to_csv(os.path.join(gaze_data_dir, f"gaze_positions_on_baby_{dt_string}.csv"))

if __name__ == "__main__":
    predictor, visualizer = get_config("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    dir = sys.argv[-1]
    baby_detection(dir, predictor, visualizer)
    
# EOF
