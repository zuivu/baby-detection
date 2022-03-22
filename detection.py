def detect_person_instance(frame, predictor, pred_threshold):
    """Finds person with high confidence prediction score (depend on the ``pred_threshold``)
    in the video frame.

    Args:
        frame (numpy.ndarray): Video frame in BGR order.
        predictor (DefaultPredictor): a simple end-to-end pre-trained predictor running on
            single device for a single input image.
        pred_threshold: Lowest accepted prediction score for the detected person instance.

    Return:
        numpy.ndarray: An array of shape (H, W), a boolean mask of the detected person.
        numpy.ndarray: An array of shape (4,), upper left and bottom right corner of the bounding
            box, in order of [start_x, start_y, end_x, end_y].
        float: Confidence score of the person prediction.

        Note: If no person is detected, the first 2 return values are arrays with no element.
    """

    outputs = predictor(frame)
    output_instances = outputs["instances"]
    person_id = predictor.metadata.thing_classes.index("person")
    person_instances = output_instances[output_instances.pred_classes == person_id]
    conf_person_instances = person_instances[person_instances.scores > pred_threshold]

    if conf_person_instances:
        conf_person_instances = conf_person_instances[0]

    return (
        conf_person_instances.pred_masks.to("cpu").numpy().squeeze(),
        conf_person_instances.pred_boxes.tensor.to("cpu").numpy().squeeze(),
        conf_person_instances.scores.item() if len(conf_person_instances.scores) else 0,
    )
