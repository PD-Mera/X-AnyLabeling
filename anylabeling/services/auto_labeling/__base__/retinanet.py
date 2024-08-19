from typing import List, Tuple, Union

from itertools import product as product
from math import ceil, sqrt

import cv2
import numpy as np

from ..engines import OnnxBaseModel

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"{{x: {self.x}, y: {self.y}}}"


class Rect(object):
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h

    def intersection_area(self, b):
        x1 = np.maximum(self.x, b.x)
        y1 = np.maximum(self.y, b.y)
        x2 = np.minimum(self.x + self.w, b.x + b.h)
        y2 = np.minimum(self.y + self.w, b.y + b.h)
        return np.abs(x1 - x2) * np.abs(y1 - y2)
    
    def __str__(self):
        return f"""{{x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}}}"""


class ObjectAttribute(object):
    def __init__(self):
        self.prob: float = 0.0
        self.rect: Rect = Rect()
        self.class_idx: int = 0
        self.use_landmark: bool = False
        self.landmark: List[Point] = []

    def __str__(self):
        if self.use_landmark:
            return f"""{{class_idx: {self.class_idx}, prob: {self.prob}, rect: {self.rect}, landmark: [{self.landmark[0]}, {self.landmark[1]}, {self.landmark[2]}, {self.landmark[3]}, {self.landmark[4]}]}}"""
        else:
            return f"""{{class_idx: {self.class_idx}, prob: {self.prob}, rect: {self.rect}}}"""


class PriorBox(object):
    def __init__(self, 
                 image_size: Union[int, Tuple[int, int]] = (640, 640),
                 min_sizes: Union[None, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None) -> None:
        super(PriorBox, self).__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.min_sizes = [[16, 32], [64, 128], [256, 512]] if min_sizes is None else min_sizes
        self.steps = [8, 16, 32]
        self.clip = True
        
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self) -> np.ndarray:
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output.clip(max=1, min=0)
        return output


class Retinanet:
    def __init__(
        self,
        onnx_model: str,
        prob_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        target_size: int = 840,
        max_size: int = 960,
        backend: str = "onnxruntime",
        device: str = "cpu",
        priorbox_min_sizes: Union[None, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.net = OnnxBaseModel(onnx_model, device_type=device)
        self.target_size = target_size
        self.max_size = max_size
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.priorbox_min_sizes = priorbox_min_sizes

    def __call__(self, image_cv2: np.ndarray) -> List[ObjectAttribute]:
        img_tensor = self.preprocess(image_cv2)
        net_outs = self.net.get_ort_inference(img_tensor, extract=False)
        dets = self.postprocess(net_outs)
        # detected_objects = self.dets2objects(dets)
        # detected_objects = self.sort_detected_objects(detected_objects, image_cv2)
        return dets

    def inference(self, blob: np.ndarray):
        """Inference model.

        Args:
            blob (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # run model
        outputs = self.net.get_ort_inference(blob)

        return outputs

    def preprocess(self, image_cv2: np.ndarray) -> np.ndarray:
        img = np.float32(image_cv2)

        # testing scale
        self.im_shape = img.shape
        im_size_min = np.min(self.im_shape[0:2])
        im_size_max = np.max(self.im_shape[0:2])
        self.resize = float(self.target_size) / float(im_size_min)
        # prevent bigger axis from being more than self.max_size:
        if np.round(self.resize * im_size_max) > self.max_size:
            self.resize = float(self.max_size) / float(im_size_max)
        if self.resize != 1:
            img = cv2.resize(img, None, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_LINEAR)
        self.im_height, self.im_width, _ = img.shape
        self.scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        self.scale_lm = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                               img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                               img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, net_outs: List[np.ndarray]) -> np.ndarray:
        """
            loc: [1, anchor, 4]
            conf: [1, anchor, 2]
            landms: [1, anchor, 10]
        """
        loc, conf, landms = net_outs
        priorbox = PriorBox(image_size=(self.im_height, self.im_width), min_sizes=self.priorbox_min_sizes)
        priors = priorbox.forward()
        prior_data = priors
        boxes = self.decode(loc.squeeze(0), prior_data, (0.1, 0.2))
        boxes = boxes * self.scale / self.resize
        # boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0)[:, 1]
        landms = self.decode_landm(landms.squeeze(0), prior_data, (0.1, 0.2))
        
        landms = landms * self.scale_lm / self.resize
        # landms = landms.cpu().numpy()

        # ignore low scores
        
        inds = np.where(scores > self.prob_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        return dets
    
    @staticmethod
    def decode(loc: np.ndarray, priors: np.ndarray, variances: Tuple[float, float]) -> np.ndarray:
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (np.ndarray): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (np.ndarray): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (Tuple[float, float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def decode_landm(pre: np.ndarray, priors: np.ndarray, variances: Tuple[float, float]) -> np.ndarray:
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (np.ndarray): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (np.ndarray): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (Tuple[float, float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), axis=1)
        return landms

    def dets2objects(self, dets: np.ndarray) -> List[ObjectAttribute]:
        """
            dets: [num_faces, 15] - x0box, y0box, x1box, y1box, conf, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5
            dets: [num_faces, 6] - x0box, y0box, x1box, y1box, conf, class_idx
        """
        detected_objects = []

        if dets.shape[1] == 15:
            for det in dets:
                detected_object = ObjectAttribute()
                detected_object.prob = det[4]
                x0 = np.maximum(np.minimum(det[0], float(self.im_shape[1]) - 1.0), 0.0)
                y0 = np.maximum(np.minimum(det[1], float(self.im_shape[0]) - 1.0), 0.0)
                x1 = np.maximum(np.minimum(det[2], float(self.im_shape[1]) - 1.0), 0.0)
                y1 = np.maximum(np.minimum(det[3], float(self.im_shape[0]) - 1.0), 0.0)
                detected_object.rect.x = x0
                detected_object.rect.y = y0
                detected_object.rect.w = x1 - x0
                detected_object.rect.h = y1 - y0
                detected_object.class_idx = 0
                detected_object.use_landmark = True
                detected_object.landmark.append(self.set_landmark_points(det[5], det[6]))
                detected_object.landmark.append(self.set_landmark_points(det[7], det[8]))
                detected_object.landmark.append(self.set_landmark_points(det[9], det[10]))
                detected_object.landmark.append(self.set_landmark_points(det[11], det[12]))
                detected_object.landmark.append(self.set_landmark_points(det[13], det[14]))
                detected_objects.append(detected_object)

        if dets.shape[1] == 6:
            for det in dets:
                detected_object = ObjectAttribute()
                detected_object.prob = det[4]
                x0 = np.maximum(np.minimum(det[0], float(self.im_shape[1]) - 1.0), 0.0)
                y0 = np.maximum(np.minimum(det[1], float(self.im_shape[0]) - 1.0), 0.0)
                x1 = np.maximum(np.minimum(det[2], float(self.im_shape[1]) - 1.0), 0.0)
                y1 = np.maximum(np.minimum(det[3], float(self.im_shape[0]) - 1.0), 0.0)
                detected_object.rect.x = x0
                detected_object.rect.y = y0
                detected_object.rect.w = x1 - x0
                detected_object.rect.h = y1 - y0
                detected_object.class_idx = int(det[5])
                detected_object.use_landmark = False
                detected_objects.append(detected_object)
            
        return detected_objects

    def sort_detected_objects(self, detected_objects: List[ObjectAttribute], image_cv2: np.ndarray) -> List[ObjectAttribute]:
        def _sort_criteria(detected_object_h_w: List[Tuple[ObjectAttribute, int, int]]) -> float:
            rect_area = detected_object_h_w[0].rect.area()
            rect_x_center = detected_object_h_w[0].rect.x + detected_object_h_w[0].rect.w / 2.0
            rect_y_center = detected_object_h_w[0].rect.y + detected_object_h_w[0].rect.h / 2.0
            img_x_center = detected_object_h_w[2] / 2.0
            img_y_center = detected_object_h_w[1] / 2.0
            dis_rect_to_center = sqrt((img_x_center - rect_x_center) ** 2 + (img_y_center - rect_y_center) ** 2)
            if dis_rect_to_center == 0:
                dis_rect_to_center = 1e-10
            return rect_area / dis_rect_to_center
        
        h, w, _ = image_cv2.shape
        detected_object_h_w = [(x, h, w) for x in detected_objects]
        detected_object_h_w = sorted(detected_object_h_w, key=_sort_criteria, reverse=True)
        detected_objects = [detected_object[0] for detected_object in detected_object_h_w]
        return detected_objects

    def set_landmark_points(self, x: float, y: float) -> Point:
        point = Point()
        point.x = x
        point.y = y
        return point
    

    @staticmethod
    def py_cpu_nms(dets: np.ndarray, thresh: float) -> np.ndarray:
        """Pure Python NMS baseline.
            dets [num_boxes, 5]
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep



