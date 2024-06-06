"""
Takes input as Frame and Returns Bounding Boxes
"""

import argparse
import os
import time
from math import ceil
import cv2
import numpy as np


# from cv2 import dnn


class FaceDetector(object):
    def __init__(self):
        self.caffe_prototxt_path = r'C:\Users\Kabra\PycharmProjects\face detection\models\RFB-320.prototxt'
        self.caffe_model_path = r'C:\Users\Kabra\PycharmProjects\face detection\models\RFB-320.caffemodel'
        # self.onnx_path = constants.configDict['onnx_path']
        self.input_siz = "320,240"
        self.threshold = 0.25
        self.image_mean = np.array([127, 127, 127])
        self.image_std = 128.0
        self.iou_threshold = 0.3
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
        self.strides = [8.0, 16.0, 32.0, 64.0]
        self.shrinkage_list = []
        self.feature_map_w_h_list = []
        # self.generate_priors = []
        # self.net = dnn.readNetFromONNX(self.onnx_path)

    def define_img_size(self, image_size):
        self.image_size = image_size
        self.shrinkage_list = []
        self.feature_map_w_h_list = []
        for self.size in self.image_size:
            self.feature_map = [int(ceil(self.size / self.stride)) for self.stride in self.strides]
            self.feature_map_w_h_list.append(self.feature_map)

        for self.i in range(0, len(self.image_size)):
            self.shrinkage_list.append(self.strides)
        self.priors = self.generate_priors(self.feature_map_w_h_list, self.shrinkage_list, self.image_size,
                                           self.min_boxes)
        return self.priors

    def generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes):
        self.feature_map_list = feature_map_list
        self.shrinkage_list = shrinkage_list
        self.image_size = image_size
        self.min_boxes = min_boxes
        self.priors = []
        for self.index in range(0, len(self.feature_map_list[0])):
            self.scale_w = self.image_size[0] / self.shrinkage_list[0][self.index]
            self.scale_h = self.image_size[1] / self.shrinkage_list[1][self.index]
            for self.j in range(0, self.feature_map_list[1][self.index]):
                for self.i in range(0, self.feature_map_list[0][self.index]):
                    self.x_center = (self.i + 0.5) / self.scale_w
                    self.y_center = (self.j + 0.5) / self.scale_h

                    for self.min_box in self.min_boxes[self.index]:
                        self.w = self.min_box / self.image_size[0]
                        self.h = self.min_box / self.image_size[1]
                        self.priors.append([
                            self.x_center,
                            self.y_center,
                            self.w,
                            self.h
                        ])
        # print("priors nums:{}".format(len(self.priors)))
        return np.clip(self.priors, 0.0, 1.0)

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        self.box_scores = box_scores
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.candidate_size = candidate_size
        self.scores = self.box_scores[:, -1]
        self.boxes = self.box_scores[:, :-1]
        self.picked = []
        self.indexes = np.argsort(self.scores)
        self.indexes = self.indexes[-self.candidate_size:]
        while len(self.indexes) > 0:
            self.current = self.indexes[-1]
            self.picked.append(self.current)
            if 0 < top_k == len(self.picked) or len(self.indexes) == 1:
                break
            self.current_box = self.boxes[self.current, :]
            self.indexes = self.indexes[:-1]
            self.rest_boxes = self.boxes[self.indexes, :]
            self.iou = self.iou_of(
                self.rest_boxes,
                np.expand_dims(self.current_box, axis=0),
            )
            self.indexes = self.indexes[self.iou <= self.iou_threshold]
        return self.box_scores[self.picked, :]

    def area_of(self, left_top, right_bottom):
        self.left_top = left_top
        self.right_bottom = right_bottom
        self.hw = np.clip(self.right_bottom - self.left_top, 0.0, None)
        return self.hw[..., 0] * self.hw[..., 1]

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        self.boxes0 = boxes0
        self.boxes1 = boxes1
        self.eps = eps
        self.overlap_left_top = np.maximum(self.boxes0[..., :2], self.boxes1[..., :2])
        self.overlap_right_bottom = np.minimum(self.boxes0[..., 2:], self.boxes1[..., 2:])

        self.overlap_area = self.area_of(self.overlap_left_top, self.overlap_right_bottom)
        self.area0 = self.area_of(self.boxes0[..., :2], self.boxes0[..., 2:])
        self.area1 = self.area_of(self.boxes1[..., :2], self.boxes1[..., 2:])
        return self.overlap_area / (self.area0 + self.area1 - self.overlap_area + self.eps)

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        self.width = width
        self.height = height
        self.confidences = confidences
        self.boxes = boxes
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.prob_threshold = prob_threshold
        self.boxes = self.boxes[0]
        self.confidences = confidences[0]
        self.picked_box_probs = []
        self.picked_labels = []
        for self.class_index in range(1, self.confidences.shape[1]):
            self.probs = self.confidences[:, self.class_index]
            self.mask = self.probs > self.prob_threshold
            self.probs = self.probs[self.mask]
            if self.probs.shape[0] == 0:
                continue
            self.subset_boxes = self.boxes[self.mask, :]
            self.box_probs = np.concatenate([self.subset_boxes, self.probs.reshape(-1, 1)], axis=1)
            self.box_probs = self.hard_nms(self.box_probs,
                                           iou_threshold=self.iou_threshold,
                                           top_k=self.top_k,
                                           )
            self.picked_box_probs.append(self.box_probs)
            self.picked_labels.extend([self.class_index] * self.box_probs.shape[0])
        if not self.picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        self.picked_box_probs = np.concatenate(self.picked_box_probs)
        self.picked_box_probs[:, 0] *= self.width
        self.picked_box_probs[:, 1] *= self.height
        self.picked_box_probs[:, 2] *= self.width
        self.picked_box_probs[:, 3] *= self.height
        return self.picked_box_probs[:, :4].astype(np.int32), np.array(self.picked_labels), self.picked_box_probs[:, 4]

    def convert_locations_to_boxes(self, locations, priors, center_variance,
                                   size_variance):
        self.locations = locations
        self.priors = priors
        self.center_variance = center_variance
        self.size_variance = size_variance
        if len(self.priors.shape) + 1 == len(self.locations.shape):
            self.priors = np.expand_dims(self.priors, 0)
        return np.concatenate([
            self.locations[..., :2] * self.center_variance * self.priors[..., 2:] + self.priors[..., :2],
            np.exp(self.locations[..., 2:] * self.size_variance) * self.priors[..., 2:]
        ], axis=len(self.locations.shape) - 1)

    def center_form_to_corner_form(self, locations):
        self.locations = locations
        return np.concatenate([self.locations[..., :2] - self.locations[..., 2:] / 2,
                               self.locations[..., :2] + self.locations[..., 2:] / 2], len(self.locations.shape) - 1)

    def maininfer(self, img_ori):
        self.img_ori = img_ori
        # self.net = dnn.readNetFromONNX(self.onnx_path)  # onnx version
        self.net = cv2.dnn.readNetFromCaffe(self.caffe_prototxt_path, self.caffe_model_path)  # caffe model converted from onnx
        self.input_size = [int(self.v.strip()) for self.v in self.input_siz.split(",")]
        self.witdh = self.input_size[0]
        self.height = self.input_size[1]
        self.priors = self.define_img_size(self.input_size)
        self.rect = cv2.resize(self.img_ori, (self.witdh, self.height))
        self.rect = cv2.cvtColor(self.rect, cv2.COLOR_BGR2RGB)
        self.net.setInput(cv2.dnn.blobFromImage(self.rect, 1 / self.image_std, (self.witdh, self.height), 127))
        self.time_time = time.time()
        self.boxes, self.scores = self.net.forward(["boxes", "scores"])
        # print("inference time: {} s".format(round(time.time() - self.time_time, 4)))
        self.boxes = np.expand_dims(np.reshape(self.boxes, (-1, 4)), axis=0)
        self.scores = np.expand_dims(np.reshape(self.scores, (-1, 2)), axis=0)
        self.boxes = self.convert_locations_to_boxes(self.boxes, self.priors, self.center_variance, self.size_variance)
        self.boxes = self.center_form_to_corner_form(self.boxes)
        self.boxes, self.labpythonels, self.probs = self.predict(self.img_ori.shape[1], self.img_ori.shape[0], self.scores,
                                                           self.boxes, self.threshold)
        #for i in range(boxes.shape[0]):
             #box = boxes[i, :]
             #cv2.rectangle(img_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        return self.boxes, self.probs
