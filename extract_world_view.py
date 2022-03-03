import os
import sys
from datetime import datetime
import cv2
from tqdm import tqdm
from utils import get_model
from gaze import get_gaze_data, save_gaze_detection, check_gaze_in_detection
from detection import detect_person
from visualize import visualize


def baby_detection(recording_dir, model_file, gaze_thres=0.8):
    """Find baby in the video and check if gaze is at the baby or not.

    :param recording_dir: directory of gaze
    :param predictor: predictor used to find baby
    :param visualizer: VideoVisualizer object
    :param gaze_thres: lowest confidence level of each gaze set by user
    """

    # Get video and its metadata
    world_vid = os.path.join(recording_dir, 'world.mp4')
    video_cap = cv2.VideoCapture(world_vid)
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    print("Frame rate:", frame_rate, "number of frames", frame_count, "width:", width, "height:", height)

    # Prepare output
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_dir = os.path.join(recording_dir, 'output', dt_string)
    os.makedirs(output_dir, exist_ok=True)
    gaze_baby_dir = os.path.join(output_dir, f"gaze_positions_on_baby_{dt_string}.csv")
    vid_dir = os.path.join(output_dir, f"world_view_with_detection_{dt_string}.avi")
    
    video_out = cv2.VideoWriter(filename=vid_dir,
                                apiPreference=cv2.CAP_ANY,
                                fourcc=cv2.VideoWriter_fourcc(*"XVID"),
                                fps=frame_rate,
                                frameSize=frame_size)

    gaze_df = get_gaze_data(recording_dir, gaze_thres, frame_count, frame_size)

    # Start processing
    detect_baby = []
    gaze_in_segment = []
    gaze_in_box = []
    predictor, viz_metadata = get_model(model_file)

    for frame_ind in tqdm(iterable=range(frame_count), desc="Processing frame", total=frame_count,
                          unit="frame", mininterval=5, miniters=1, dynamic_ncols=True):

        retval, frame = video_cap.read()
        if retval:
            # Get gazes
            gaze_datum = gaze_df.loc[gaze_df['world_index'] == frame_ind]

            # Detectron algo 
            segmentation, bounding_box, pred_score = detect_person(frame, predictor)
            num_gaze = len(gaze_datum)
            if pred_score:
                detect_baby.extend([True]*num_gaze)
            else:
                detect_baby.extend([False]*num_gaze)

            # Check gaze in detection
            all_gaze_pos = gaze_datum["world_coord"].to_list()     # get_gaze_in_frame(gaze_datum, width, height)
            all_gaze_status = []
            for gaze_pos in all_gaze_pos:
                in_segmentation, in_box, gaze_status = check_gaze_in_detection(gaze_pos, segmentation, bounding_box)
                gaze_in_segment.append(in_segmentation)
                gaze_in_box.append(in_box)
                all_gaze_status.append(gaze_status)

            viz_frame = visualize(frame=frame,
                                  segmentation=segmentation,
                                  pred_score=pred_score,
                                  gaze_list=zip(all_gaze_pos, all_gaze_status),
                                  viz_metadata=viz_metadata)

            video_out.write(viz_frame)

            if cv2.waitKey(delay=1) & 0xFF == ord('q'):
                sys.exit()

    video_cap.release()
    video_out.release()

    save_gaze_detection(gaze_df, detect_baby, gaze_in_segment, gaze_in_box, gaze_baby_dir)
    

if __name__ == "__main__":
    model_config_file = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    dir = sys.argv[-1]  # "public_data/2022-02-26-23-45-57"
    baby_detection(dir, model_config_file)
    
# EOF