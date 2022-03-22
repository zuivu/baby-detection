import os
import sys
from datetime import datetime
import cv2
from tqdm import tqdm
from utils import get_model
from gaze import get_gaze_data, check_gaze_in_detection, save_gaze_detection
from detection import detect_person_instance
from visualize import visualize


def detect_baby(recording_dir, model_file, min_detection_score=0.93):
    """Create new video with visualization of detected baby and gazes.

    Arg:
        recording_dir (str): Directory of exported recording from Pupil Player.
        config_file (str): Configuration file.
        min_detection_score (float): Lowest accepted prediction score for the
        detected person instance. Defaults to 0.93.
    """

    # Get video and its metadata
    world_vid = os.path.join(recording_dir, "world.mp4")
    video_cap = cv2.VideoCapture(world_vid)
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    print(
        f"Frame rate: {round(frame_rate,2)} fps, "
        + f"total frames: {frame_count}, "
        + f"width: {width}, height: {height}."
    )

    # Prepare output
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_dir = os.path.join(recording_dir, "output", dt_string)
    os.makedirs(output_dir, exist_ok=True)
    gaze_baby_dir = os.path.join(output_dir, f"gaze_positions_on_baby_{dt_string}.csv")
    vid_dir = os.path.join(output_dir, f"world_view_with_detection_{dt_string}.avi")

    # Start processing
    detect_baby = []
    gaze_in_segment = []
    gaze_in_box = []

    predictor = get_model(model_file)
    gaze_df = get_gaze_data(recording_dir, frame_count, frame_size)

    video_out = cv2.VideoWriter(
        filename=vid_dir,
        apiPreference=cv2.CAP_ANY,
        fourcc=cv2.VideoWriter_fourcc(*"XVID"),
        fps=frame_rate,
        frameSize=frame_size,
    )

    for frame_ind in tqdm(
        iterable=range(frame_count),
        desc="Processing frame",
        total=frame_count,
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
            all_gaze_pos = gaze_datum["world_coord"].to_list()
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
    # new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py  accurate and fast 43.5 AP, 0.071 s
    # new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py  most accurate 43.7 AP, 0.073 s
    # "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" most accurate 39.5 AP, 0.103 s
    # "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" accurate and fast 38.6 AP, 0.056 s
    # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" fastest 0.043 AP, 37.2 s
    model_config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    dir = sys.argv[-1]  # "public_data/2022-02-26-23-45-57"
    detect_baby(dir, model_config_file)
