import os
import sys
from datetime import datetime
import cv2
from tqdm import tqdm
from utils import get_model, get_clicking_frame_timeline
from gaze import get_gaze_data, check_gaze_in_detection, save_gaze_detection
from detection import detect_person_instance
from visualize import visualize
import toml


def detect_baby(
    recording_dir, model_file, frame_duration, output_dir, suffix_out_dir, min_detection_score=0.9
):
    """Create new video with visualization of detected baby and gazes.

    Arg:
        recording_dir (str): Directory of exported recording from Pupil Player.
        config_file (str): Configuration file.
        frame_duration (tuple of int): Start and end frame of the experiment.
        output_dir (str): Directory of output folder.
        suffix_out_dir (str): Suffix added to the names of video and csv files in the output folder.
        min_detection_score (float): Lowest accepted prediction score for the
            detected person instance. Defaults to 0.93.
    """

    # Get video and its metadata
    world_vid = os.path.join(recording_dir, "world.mp4")
    video_cap = cv2.VideoCapture(world_vid)
    frame_start, frame_end = frame_duration
    frame_count = frame_end - frame_start
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(
        f"Frame rate: {round(frame_rate,2)} fps, "
        + f"total frames: {frame_count}, "
        + f"width: {frame_size[0]}, height: {frame_size[1]}."
    )

    # Prepare output
    os.makedirs(output_dir, exist_ok=True)
    gaze_baby_dir = os.path.join(output_dir, f"gaze_positions_on_baby_{suffix_out_dir}.csv")
    vid_dir = os.path.join(output_dir, f"world_view_with_detection_{suffix_out_dir}.avi")

    # Start processing
    detect_baby = []
    gaze_in_segment = []
    gaze_in_box = []

    predictor = get_model(model_file)
    gaze_df = get_gaze_data(recording_dir, frame_duration, frame_size)

    video_out = cv2.VideoWriter(
        filename=vid_dir,
        apiPreference=cv2.CAP_ANY,
        fourcc=cv2.VideoWriter_fourcc(*"XVID"),
        fps=frame_rate,
        frameSize=frame_size,
    )

    for frame_ind in tqdm(
        iterable=range(frame_start, frame_end),
        desc="Processing frame",
        unit="frame",
        mininterval=5,
        miniters=1,
        dynamic_ncols=True,
    ):
        retval, frame = video_cap.read()
        if retval:
            # Get gazes
            gaze_datum = gaze_df.loc[gaze_df["world_index"] == frame_ind]

            # Detectron algo
            segmentation, bounding_box, pred_score = detect_person_instance(
                frame, predictor, min_detection_score
            )
            num_gaze = len(gaze_datum)
            if pred_score:
                detect_baby.extend([True] * num_gaze)
            else:
                detect_baby.extend([False] * num_gaze)

            # Check gaze in detection
            all_gaze_pos = gaze_datum.loc[:, "world_pos"].to_list()
            all_gaze_status = []
            for gaze_pos in all_gaze_pos:
                in_segmentation, in_box, gaze_status = check_gaze_in_detection(
                    gaze_pos, segmentation, bounding_box
                )
                gaze_in_segment.append(in_segmentation)
                gaze_in_box.append(in_box)
                all_gaze_status.append(gaze_status)

            viz_frame = visualize(
                frame=frame,
                segmentation=segmentation,
                pred_score=pred_score,
                gaze_list=zip(all_gaze_pos, all_gaze_status),
            )

            video_out.write(viz_frame)

            if cv2.waitKey(delay=1) & 0xFF == ord("q"):
                sys.exit()

    video_cap.release()
    video_out.release()

    save_gaze_detection(gaze_df, detect_baby, gaze_in_segment, gaze_in_box, gaze_baby_dir)


if __name__ == "__main__":
    config = toml.load("config.toml")
    seed = config.get("seed_number")
    set_all_seeds(seed)
    model_config_file = config.get("model").get("model_config_path")
    eye_tracking_dir = config.get("data").get("data_directory")

    output_dir = os.path.join(
        eye_tracking_dir, "output", datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    )
    frame_timeline = get_clicking_frame_timeline(os.path.join(eye_tracking_dir, "end_frames.txt"))

    print(eye_tracking_dir)
    print(model_config_file)
    print(output_dir)
    print(frame_timeline)
    print()

    for exp_id, frame_duration in enumerate(frame_timeline):
        detect_baby(
            recording_dir=eye_tracking_dir,
            model_file=model_config_file,
            frame_duration=frame_duration,
            output_dir=output_dir,
            suffix_out_dir=f"part_{exp_id+1}",
        )
