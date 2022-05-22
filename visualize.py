import cv2
from detectron2.utils.visualizer import Visualizer
from gaze import Gaze


def visualize(frame, segmentation, pred_score, gaze_list, show=False):
    """Visualizes result on video frame.

    Args:
        frame (numpy.ndarray): Video frame in BGR order.
        segmentation (numpy.ndarray): An array of shape (H, W), a boolean mask of the detected
            person. Or an array with no element if no person is detected.
        pred_score (float): Confidence score of the person prediction.
        gaze_list (list of tuple): List of gaze coordinates on the frame and its status (in
            detection or not).
        show (bool): True to show the result visualization frame. False otherwises.

    Return:
        numpy.ndarray: Video frame in BGR order with segmentation, and gaze points drawn on.
    """

    gaze_bgr = {
        Gaze.IN_DETECTION: (0, 255, 0),  # green
        Gaze.NOT_IN_DETECTION: (0, 0, 255),  # red
        Gaze.NO_DETECTION: (0, 255, 255),  # yellow
    }

    if pred_score:
        out = (
            Visualizer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            .draw_binary_mask(
                segmentation,
                color="cornflowerblue",  # "mediumblue"
                edge_color="blue",
                # text=f"baby {round(pred_score*100, 1)}%",
                alpha=0.4,
            )
            .get_image()
        )

        frame = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    for gaze_pos, gaze_status in gaze_list:
        if isinstance(gaze_pos, tuple):
            frame = cv2.circle(
                img=frame, center=gaze_pos, radius=3, color=gaze_bgr[gaze_status], thickness=-1
            )

    if show:
        cv2.imshow("Recording", frame)

    return frame
