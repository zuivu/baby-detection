import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
from gaze import Gaze

def visualize(frame, segmentation, pred_score: float, gaze_list, viz_metadata, show=False):
    """Visualize result from predictor

    :param frame: BGR video frame with shape (H, W, 3)
    :type frame: numpy.ndarray
    :param pred_instance: Instance object of a prediction result from predictor
    :type pred_instance: detectron2.structures.Instances
    :param visualizer: Visualizer objects
    :type visualizer: detectron2.utils.video_visualizer.VideoVisualizer

    :return: Segmentation, bounding box of the baby, and the gaze point in the
    frame in BGR ordering.
    :rtype: numpy.ndarray
    """
    gaze_bgr = {
        Gaze.IN_DETECTION: (0, 255, 0),         # green
        Gaze.NOT_IN_DETECTION: (0, 0, 255),     # red
        Gaze.NO_DETECTION: (0, 255, 255)        # yellow
    }

    if pred_score:
        out = Visualizer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                        metadata=viz_metadata,
                        instance_mode=ColorMode.SEGMENTATION
        ).draw_binary_mask(segmentation, 
                        color="blue",
                        edge_color="mediumblue",
                        text=f"baby {round(pred_score*100, 1)}%",
                        alpha=0.4
        ).get_image()

        frame = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    for gaze_pos, gaze_status in gaze_list:
        if isinstance(gaze_pos, tuple):
            frame = cv2.circle(img=frame, 
                               center=gaze_pos, 
                               radius=3, 
                               color=gaze_bgr[gaze_status],
                               thickness=-1)
    
    if show: 
        cv2.imshow("Recording", frame)

    return frame
