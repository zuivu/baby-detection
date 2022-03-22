from torch import cuda
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def get_model(config_file):
    """Returns detection model.

    Arg:
        config_file (str): Configuration file.

    Return:
        DefaultPredictor: End-to-end predictor object with the given config that runs on single
            device for a single input image.
    """

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model, faster inference
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

    if cuda.is_available():
        print(f"Run on CUDA: {cuda.get_device_name(0)}.", end="\n\n")
    else:
        print("CUDA is not available.", end="\n\n")
        cfg.MODEL.DEVICE = "cpu"

    return DefaultPredictor(cfg)
