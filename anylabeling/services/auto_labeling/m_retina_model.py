import logging

from itertools import product as product

import os
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from .model import Model
from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .__base__.retinanet import Retinanet


class RetinaModel(Model):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = ["button_run"]
        output_modes = {
            "point": QCoreApplication.translate("Model", "Point"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(
            self.config, "model_path"
        )
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize Retina model."
                )
            )

        self.draw_det_box = self.config.get("draw_det_box", True)
        self.nms_threshold = self.config.get("nms_threshold", 0.65)
        self.target_size = self.config.get("target_size", 640)
        self.max_size = self.config.get("max_size", 720)
        self.priorbox_min_sizes = self.config.get("priorbox_min_sizes", None)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.45)
        self.kpt_classes = self.config.get("kpt_classes", [])
        self.retinanet = Retinanet(model_abs_path, 
                                   prob_threshold = self.confidence_threshold,
                                   nms_threshold = self.nms_threshold,
                                   target_size = self.target_size,
                                   max_size = self.max_size,
                                   priorbox_min_sizes = self.priorbox_min_sizes)

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        det_results = self.retinanet(image)

        shapes = []
        for i, det in enumerate(det_results):
            # x1, y1, x2, y2 = list(map(int, bbox))
            x0 = int(np.maximum(np.minimum(det[0], float(self.retinanet.im_shape[1])), 0.0))
            y0 = int(np.maximum(np.minimum(det[1], float(self.retinanet.im_shape[0])), 0.0))
            x1 = int(np.maximum(np.minimum(det[2], float(self.retinanet.im_shape[1])), 0.0))
            y1 = int(np.maximum(np.minimum(det[3], float(self.retinanet.im_shape[0])), 0.0))

            if self.draw_det_box:
                rectangle_shape = Shape(
                    label="license_plate", shape_type="rectangle", group_id=int(i)
                )
                rectangle_shape.add_point(QtCore.QPointF(x0, y0))
                rectangle_shape.add_point(QtCore.QPointF(x1, y0))
                rectangle_shape.add_point(QtCore.QPointF(x1, y1))
                rectangle_shape.add_point(QtCore.QPointF(x0, y1))
                shapes.append(rectangle_shape)

            img = image[y0:y1, x0:x1]

            for j in range(5):
                point_shape = Shape(
                    label=str(self.kpt_classes[j]),
                    shape_type="point",
                    group_id=int(i),
                )
                point_shape.add_point(QtCore.QPointF(int(det[5 + j * 2]), int(det[5 + j * 2 + 1])))
                shapes.append(point_shape)


        result = AutoLabelingResult(shapes, replace=True)

        return result

    def unload(self):
        del self.retinanet
