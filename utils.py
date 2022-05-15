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

    device = "cpu"
    if cuda.is_available():
        print(f"Run on CUDA: {cuda.get_device_name(0)}.", end="\n\n")
        device = "cuda"
    else:
        print("CUDA is not available.", end="\n\n")

    if config_file.endswith(".yaml"):  # Support old version of predictor
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            0.7  # set threshold for this model, faster inference
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        cfg.MODEL.DEVICE = device
        if cuda.is_available():
            cfg.MODEL.DEVICE = device

        return DefaultPredictor(cfg)

    # Newer version
    # TODO:
    # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultPredictor
    # MetadataCatalog.get(cfg2.dataloader.test.dataset.names)
    # def __call__(self, original_image):
    model = model_zoo.get(config_file, trained=True, device=device)
    return model


def get_clicking_frame_timeline(clicking_frames_dir):
    """Get frames where clicking sounds (signal begin and end of experiment) happen.

    Arg:
        clicking_frames_dir (str): Directory of exported recording from Pupil Player.

    Return:
        list of tuple of int: List of experiments' start and end frames.

    Note: Tuple is prefered to store frames since they are immutable while list is not.
    """

    frames_end = []
    with open(clicking_frames_dir, "r") as frames_file:
        for frames in frames_file:
            frames_end.append(tuple([int(frame_id) for frame_id in frames.split(";")]))
    return frames_end
