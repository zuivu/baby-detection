import os
from enum import Enum
from warnings import warn

import numpy as np
import pandas as pd


class Gaze(Enum):
    IN_DETECTION = "gaze_in_detection",
    NOT_IN_DETECTION = "gaze_not_in_detection",
    NO_DETECTION = "person_not_detected"


def filling_missing_gaze(gaze_df, total_frame):
    """Add rows of gaze empty data where world indices are missing from the current dataframe.
    Rows added has NA values for all coordinate data and zero confidence for gaze.

    Args:
        gaze_df (pandas.Dataframe): Gaze dataframe with missing world indices.
        total_frame (int): Number of video frames.

    Return:
        pandas.Dataframe: Dataframe with all of world indices.
    """
    
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
    """Gets fully processed gaze dataframe from the recording directory. 

    Args:
        recording_dir (str): Directory of exported recording from Pupil Player.
        gaze_thres (float): Lowest accepted confidence level for gaze.
        total_frame (int): Number of video frames.
        frame_size (tuple of int): Width and height of the video frame.

    Return:
        pandas.Dataframe: Gaze dataframe without missing gaze data and gaze coordinates
            corresponding to video frame. 
    """
    
    gaze_data_dir = os.path.join(recording_dir, 'exports', os.listdir(os.path.join(recording_dir, 'exports'))[0])
    raw_gaze_df = pd.read_csv(os.path.join(gaze_data_dir, 'gaze_positions.csv'), usecols=["world_index", "confidence", "norm_pos_x", "norm_pos_y"])

    min_thres = get_gaze_thres(raw_gaze_df, total_frame)
    if gaze_thres > min_thres:
        warn(f"Use threshold smaller than or equal to {min_thres:.2f} to have at least one gaze per frame.")

    # gaze_df = raw_gaze_df.loc[raw_gaze_df['confidence'] >= gaze_thres]
    unused_gaze_pct = (1 - sum(raw_gaze_df['confidence'] >= gaze_thres)/len(raw_gaze_df)) * 100
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
    """Save gaze dataframe to disk.
    
    Args:
        gaze_df (pandas.Dataframe): Gaze dataframe.
        detect_baby (list of bool): Baby is detected or not on each video frame.  
        gaze_in_segment (list of bool|NA): Gaze is in person segmentation or not (NA if no detected baby) on each video frame.
        gaze_in_box (list of bool|NA): Gaze is in person bounding box or not (NA if no detected baby) on each video frame.
        gaze_baby_dir (str): File directory where data is saved.  
    """
    
    gaze_df["is_baby"] = detect_baby
    gaze_df["in_segmentation"] = gaze_in_segment
    gaze_df["in_bounding_box"] = gaze_in_box
    gaze_df.to_csv(gaze_baby_dir)


def check_gaze_in_detection(gaze_pos, mask, box):
    """Check if gaze point is in segmentation, bounding box and assign a status for each gaze:
    
    1. Gaze in segmentation of detected person.
    2. Gaze not segmentation of detected person.
    3. No detected person.

    Args:
        gaze_pos (tuple of int): Coordinate of gaze in video frame.
        mask (numpy.ndarray): An array of shape (H, W), a boolean mask of the detected person.
        box (numpy.ndarray): An array of shape (4,), upper left and bottom right corner of the bounding box,
            in order of [start_x, start_y, end_x, end_y].
        
        Note: If no person is detected, ``mask`` and ``box`` are arrays with no element.

    Returns:
        If no person is detected, np.nan for the first 2 return values. Otherwise,
            1. bool: Whether gaze is in segmentation.
            2. bool: Whether gaze is in bounding box.

        3. enum 'Gaze': 1 of 3 gaze status defined in enum 'Gaze'.
    """

    if not (len(mask) or len(box)):
        return np.nan, np.nan, Gaze.NO_DETECTION

    if pd.isna(gaze_pos):
        in_segment = in_box = False
    else:
        gaze_x, gaze_y = gaze_pos
        in_segment = mask[gaze_y, gaze_x]

        (start_x, start_y), (end_x, end_y) = np.floor(box[:2]), np.ceil(box[2:])
        in_box = (start_x <= gaze_x <= end_x) and (start_y <= gaze_y <= end_y)

    gaze_status = Gaze.IN_DETECTION if in_segment else Gaze.NOT_IN_DETECTION

    return in_segment, in_box, gaze_status

# Not in use anymore
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
