import os
from enum import Enum
from warnings import warn

import numpy as np
import pandas as pd


class Gaze(Enum):
    IN_DETECTION = "gaze_in_detection",
    NOT_IN_DETECTION = "gaze_not_in_detection",
    NO_DETECTION = "person_not_detected"


def get_gaze_thres(gaze_df, total_frame):
    """Get the lowest threshold of confidence user can set so that there is at
    least one gaze point in each frame
    
    :param gaze_df: dataframe containing gaze info
    :param total_frame: number of video's frames

    :return threshold: lowest threshold
    :rtype: float
    """

    threshold_list = gaze_df['confidence'].drop_duplicates().sort_values()
    for threshold in threshold_list:
        good_gaze_df = gaze_df.loc[gaze_df['confidence'] > threshold] 
        if good_gaze_df['world_index'].nunique() < total_frame:
            return threshold


def filling_missing_gaze(gaze_df, total_frame):
    missing_world_indices = list(set(range(total_frame)) - set(gaze_df["world_index"]))
    missing_gaze_df = pd.DataFrame(
        {
            "world_index": missing_world_indices, 
            "confidence": [0]*len(missing_world_indices),
        }
    )
    full_gaze_df = pd.concat([missing_gaze_df, gaze_df], ignore_index=True)
    full_gaze_df.sort_values(by=["world_index"], ignore_index=True, inplace=True)
    
    return full_gaze_df


def get_gaze_data(recording_dir, gaze_thres, total_frame, frame_size):
    gaze_data_dir = os.path.join(recording_dir, 'exports', os.listdir(os.path.join(recording_dir, 'exports'))[0])
    raw_gaze_df = pd.read_csv(os.path.join(gaze_data_dir, 'gaze_positions.csv'), usecols=["world_index", "confidence", "norm_pos_x", "norm_pos_y"])

    min_thres = get_gaze_thres(raw_gaze_df, total_frame)
    if gaze_thres > min_thres:
        warn(f"Use threshold smaller than or equal to {min_thres:.2f} to have at least one gaze per frame.")

    # gaze_df = raw_gaze_df.loc[raw_gaze_df['confidence'] >= gaze_thres]
    unused_gaze_pct = (1 - sum(raw_gaze_df['confidence'] >= gaze_thres) /len(raw_gaze_df)) * 100
    print(f"{round(unused_gaze_pct)}% of gaze points will not be used due to low confidence (< {gaze_thres}).")

    def cal_gaze_in_frame(x_norm, y_norm):
        width, height = frame_size
        world_x, world_y = np.clip(a=[round(x_norm * width), round((1-y_norm) * height)],
                                   a_min=[0, 0], 
                                   a_max=[width - 1, height - 1]
                                  )
        return (world_x, world_y)

    raw_gaze_df["world_coord"] = raw_gaze_df.apply(
        lambda row: cal_gaze_in_frame(row["norm_pos_x"], row["norm_pos_x"]),
        axis=1
    )

    final_gaze_df = filling_missing_gaze(raw_gaze_df, total_frame)

    return final_gaze_df


def save_gaze_detection(gaze_df, detect_baby, gaze_in_segment, gaze_in_box, gaze_baby_dir):
    #print(gaze_df, detect_baby, gaze_in_segment, gaze_in_box, gaze_baby_dir)
    gaze_df["is_baby"] = detect_baby
    gaze_df["in_segmentation"] = gaze_in_segment
    gaze_df["in_bounding_box"] = gaze_in_box
    gaze_df.to_csv(gaze_baby_dir)


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


def check_gaze_in_detection(gaze_pos, mask, box):
    """Check if gaze point (calculated) is within segmentation and bounding box
    or not and assign gaze's color in each case (green - face is detected and 
    gaze is in face's bounding box, red - face is detected but gaze is not in 
    face's box, yellow - face is not detected)

    :param gaze_pos: coordinate of gaze
    :param mask: bitmask for the detected baby, same shape with video frame 
    :param box: coordinates of upper left and bottom right corners of the box

    :return in_segment: whether gaze is in segmentation or not
    :rtype: boolean
    :return in_box: whether gaze is in bounding box or not
    :rtype: boolean
    :return gaze_color: RGB value for the gaze
    :rtype: tuple
    """

    if not (len(mask) or len(box)):
        return np.nan, np.nan, Gaze.NO_DETECTION

    # gaze_x, gaze_y = np.clip(a=gaze_pos, a_min=0, a_max=(mask.shape[1] - 1, mask.shape[0] - 1))
    
    if pd.isna(gaze_pos):
        in_segment = in_box = False
    else:
        gaze_x, gaze_y = gaze_pos
        in_segment = mask[gaze_y, gaze_x]

        (start_x, start_y), (end_x, end_y) = np.floor(box[:2]), np.ceil(box[2:])
        in_box = (start_x <= gaze_x <= end_x) and (start_y <= gaze_y <= end_y)

    gaze_status = Gaze.IN_DETECTION if in_segment else Gaze.NOT_IN_DETECTION

    return in_segment, in_box, gaze_status
