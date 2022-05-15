import os
from enum import Enum
from warnings import warn

import numpy as np
import pandas as pd


class Gaze(Enum):
    IN_DETECTION = "gaze_in_detection"
    NOT_IN_DETECTION = "gaze_not_in_detection"
    NO_DETECTION = "person_not_detected"


def get_gaze_thres(gaze_df: pd.DataFrame, total_frame: int):
    """Get the lowest threshold of confidence user can set so that there is at
    least one gaze point in each frame

    :param gaze_df: dataframe containing gaze info
    :param total_frame: number of video's frames

    :return threshold: lowest threshold
    :rtype: float
    """

    threshold_list = gaze_df["confidence"].drop_duplicates().sort_values()
    for threshold in threshold_list:
        good_gaze_df = gaze_df.loc[gaze_df["confidence"] > threshold]
        if good_gaze_df.loc[:, "world_index"].nunique() < total_frame:
            return threshold


def clipping_gaze(denormalized_gazes: pd.Series, frame_dimension: int):
    """Returns gazes in denormalized coordinate (float numbers) to the indicies on the video frame.

    Args:
        denormalized_gazes (Series): Gazes in denormalized coordinate.
        frame_dimension (int): Dimension of the frame to clip the coordinates to avoid index out of
            range.

    Return:
        list: List of gaze coordinates that can be used as indicies on the video frame.
    """

    return (
        np.clip(a=np.rint((denormalized_gazes - 1).to_numpy()), a_min=0, a_max=frame_dimension - 1)
        .astype(int)
        .tolist()
    )


def map_gaze_to_frame_coord(gaze_data: pd.DataFrame, frame_size: tuple):
    """Converts gazes in normalized coordinate to video coordinate.

    Args:
        gaze_data (Dataframe): Gaze dataframe.
        frame_size (tuple of int): Width and height of the video frame.

    Return:
        np.ndarray: Array of tuple of gaze coordinates.
    """

    width, height = frame_size
    x_denormalized = gaze_data.loc[:, "norm_pos_x"] * width
    y_denormalized = (1 - gaze_data.loc[:, "norm_pos_y"]) * height
    x_world_coor = clipping_gaze(x_denormalized, width)
    y_world_coor = clipping_gaze(y_denormalized, height)

    return np.array(zip(x_world_coor, y_world_coor))


def filling_missing_gaze(gaze_df: pd.DataFrame, frame_duration: int):
    """Add rows of gaze empty data where world indices are missing from the current dataframe.
    Rows added has NA values for all coordinate data and zero confidence for gaze.

    Args:
        gaze_df (pandas.Dataframe): Gaze dataframe with missing world indices.
        total_frame (int): Number of video frames.

    Return:
        pandas.Dataframe: Dataframe with all of world indices.
    """
    all_frames_range = range(frame_duration[0], frame_duration[1])
    missing_frame_indices = list(set(all_frames_range) - set(gaze_df["world_index"]))
    missing_gaze_df = pd.DataFrame(
        {
            "world_index": missing_frame_indices,
            "confidence": [0] * len(missing_frame_indices),
        }
    )
    full_gaze_df = pd.concat([missing_gaze_df, gaze_df], ignore_index=True)
    full_gaze_df.sort_values(by=["world_index"], ignore_index=True, inplace=True)

    return full_gaze_df


def get_gaze_data(
    recording_dir: str, frame_duration: int, frame_size: int, gaze_thres: float = 0.8
):
    """Gets fully processed gaze dataframe from the recording directory.

    Args:
        recording_dir (str): Directory of exported recording from Pupil Player.
        frame_duration (tuple of int): Start and end frame of the experiment.
        frame_size (tuple of int): Width and height of the video frame.
        gaze_thres (float): Lowest accepted confidence level for gaze. Defaults to 0.8

    Return:
        pandas.Dataframe: Gaze dataframe without missing gaze data and gaze coordinates
            corresponding to video frame.
    """

    gaze_data_dir = os.path.join(
        recording_dir, "exports", os.listdir(os.path.join(recording_dir, "exports"))[0]
    )
    raw_gaze_df = pd.read_csv(
        os.path.join(gaze_data_dir, "gaze_positions.csv"),
        usecols=["world_index", "confidence", "norm_pos_x", "norm_pos_y"],
    )

    frame_start, frame_end = frame_duration
    raw_gaze_df = raw_gaze_df.loc[
        (frame_start <= raw_gaze_df["world_index"]) & (raw_gaze_df["world_index"] < frame_end)
    ]

    min_thres = get_gaze_thres(raw_gaze_df, total_frame=frame_end - frame_start)
    if gaze_thres > min_thres:
        warn(
            f"Use threshold smaller than or equal to {min_thres:.1f} to retain at least one gaze "
            + "per frame."
        )

    good_gaze_df = raw_gaze_df.loc[raw_gaze_df["confidence"] >= gaze_thres]
    unused_gaze_pct = round((1 - len(good_gaze_df) / len(raw_gaze_df)) * 100)
    if unused_gaze_pct:
        print(f"{unused_gaze_pct}% of gazes are removed due to low confidence (< {gaze_thres}).")

    good_gaze_df.loc[:, "world_pos"] = map_gaze_to_frame_coord(good_gaze_df, frame_size)
    full_gaze_df = filling_missing_gaze(good_gaze_df, frame_duration)

    return full_gaze_df


def save_gaze_detection(
    gaze_df: pd.DataFrame,
    detect_baby: list,
    gaze_in_segment: list,
    gaze_in_box: list,
    gaze_baby_dir: str,
):
    """Save gaze dataframe to disk.

    Args:
        gaze_df (pandas.Dataframe): Gaze dataframe.
        detect_baby (list of bool): Baby is detected or not on each video frame.
        gaze_in_segment (list of bool|NA): Gaze is in person segmentation or not (NA if no detected
            baby) on each video frame.
        gaze_in_box (list of bool|NA): Gaze is in person bounding box or not (NA if no detected
            baby) on each video frame.
        gaze_baby_dir (str): File directory where data is saved.
    """

    gaze_df.loc[:, ["is_baby", "in_segmentation", "in_bounding_box"]] = np.column_stack(
        (detect_baby, gaze_in_segment, gaze_in_box)
    )
    # gaze_df["in_segmentation"] = gaze_in_segment
    # gaze_df["in_bounding_box"] = gaze_in_box
    gaze_df.to_csv(gaze_baby_dir, index=False)


def check_gaze_in_detection(gaze_pos, mask, box):
    """Check if gaze point is in segmentation, bounding box and assign a status for each gaze:

    1. Gaze in segmentation of detected person.
    2. Gaze not segmentation of detected person.
    3. No detected person.

    Args:
        gaze_pos (tuple of int): Coordinate of gaze in video frame.
        mask (numpy.ndarray): An array of shape (H, W), a boolean mask of the detected person.
        box (numpy.ndarray): An array of shape (4,), upper left and bottom right corner of the
            bounding box, in order of [start_x, start_y, end_x, end_y].

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
