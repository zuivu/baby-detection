def detect_person(im, predictor, pred_threshold=0.93):
    """Find persons in the image

    :param im: Image in the format of BGR with shape (H, W, 3)
    :type im: numpy.ndarray
    :param predictor: DefaultPredictor object 
    :type predictor: predictor object

    :return baby_instance: Instance object containing prediction's attributes
    of all person
    :rtype: detectron2.structures.Instances
    """

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    outputs = predictor(im)
    output_instances = outputs["instances"]
    person_id = predictor.metadata.thing_classes.index('person')
    person_instances = output_instances[output_instances.pred_classes == person_id]
    conf_person_instances = person_instances[person_instances.scores > pred_threshold]
    
    if conf_person_instances:
        conf_person_instances = conf_person_instances[0]

    return conf_person_instances.pred_masks.to("cpu").numpy().squeeze(), \
           conf_person_instances.pred_boxes.tensor.to("cpu").numpy().squeeze(), \
           conf_person_instances.scores.item() if len(conf_person_instances.scores) else 0
