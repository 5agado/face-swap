import logging
from ast import literal_eval

import cv2
import dlib
import numpy as np

from face_swap import faceswap_utils as utils
from face_swap.Face import Face


class FaceSwapException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class FaceDetector:
    # constant locators for landmarks
    jaw_points = np.arange(0, 17) # face contour points
    eyebrow_dx_points = np.arange(17, 22)
    eyebrow_sx_points = np.arange(22, 27)
    nose_points = np.arange(27, 36)
    nosecenter_points = np.array([30, 33])
    right_eye = np.arange(36, 42)
    left_eye = np.arange(42, 48)

    def __init__(self, config):
        self.config = config

        # if specified use mtcnn model
        self.mtcnn_model_path = config.get('mtcnn_model_path', None)
        if self.mtcnn_model_path:
            from mtcnn.mtcnn import MTCNN
            self.detector = MTCNN()
        # otherwise rely on dlib detector
        else:
            self.detector_model_path = config.get('detector_path', None)
            if self.detector_model_path:
                self.detector = dlib.cnn_face_detection_model_v1(self.detector_model_path)
            else:
                self.detector = dlib.get_frontal_face_detector()
        # always instantiate predictor
        self.predictor = dlib.shape_predictor(config.get('shape_predictor_path'))

    def _mtcnn_detect_faces(self, img):
        face_confidence_threshold = 0.9
        rects = self.detector.detect_faces(img)
        faces = [Face(img.copy(), None,
                      # output bbox coordinate of MTCNN is [x, y, width, height]
                      # need to max to 0 cause sometimes bbox has negative values ??library BUG
                      dlib.rectangle(left=max(r['box'][0], 0), top=max(r['box'][1], 0),
                                     right=max(r['box'][0], 0) + max(r['box'][2], 0),
                                     bottom=max(r['box'][1], 0) + max(r['box'][3], 0)))
                 for r in rects if r['confidence'] > face_confidence_threshold]

        return faces

    def detect_faces(self, img):
        if self.mtcnn_model_path:
            faces = self._mtcnn_detect_faces(img)
        else:
            rects = self.detector(img, 1)
            # if using custom detector we need to extract get the rect attribute
            if self.detector_model_path:
                rects = [r.rect for r in rects]
            faces = [Face(img.copy(), None, r) for r in rects]

        # continue only if we detected at least one face
        if len(faces) == 0:
            logging.debug("No face detected")
            raise FaceSwapException("No face detected.")
        for f in faces:
            # border expansion causes error during swap process
            f.face_img = self._extract_face(f, out_size=None,
                                            border_expand=(0, 0),  #self.config['extract']['border_expand'],
                                            align=False)
        return faces

    def get_landmarks(self, face: Face, recompute=False):
        # If landmarks already present, just return, unless is required to recompute them
        if face.landmarks is not None and not recompute:
            return face.landmarks
        else:
            shape = self.predictor(face.img, face.rect)
            return np.array([(p.x, p.y) for p in shape.parts()])

    def get_contour(self, face: Face):
        shape = self.predictor(face.img, face.rect)
        return self.get_contour_points(shape)

    @staticmethod
    def get_eyes(face: Face):
        lx_eye = face.landmarks[FaceDetector.left_eye]
        rx_eye = face.landmarks[FaceDetector.right_eye]
        return lx_eye, rx_eye

    @staticmethod
    def get_contour_points(shape):
        # shape to numpy
        points = np.array([(p.x, p.y) for p in shape.parts()])
        face_boundary = points[np.concatenate([FaceDetector.jaw_points,
                                               FaceDetector.eyebrow_dx_points,
                                               FaceDetector.eyebrow_sx_points])]
        return face_boundary, shape.rect

    def extract_face(self, face: Face):
        # size is a tuple, so need to eval from string
        # representation in config
        size = literal_eval(self.config['extract']['size'])
        border_expand = literal_eval(self.config['extract']['border_expand'])
        align = self.config['extract']['align']
        maintain_proportion = self.config['extract']['maintain_proportion']
        masked = self.config['extract']['masked']

        return self._extract_face(face, size, border_expand=border_expand, align=align,
                                  maintain_proportion=maintain_proportion,
                                  masked=masked)

    def _extract_face(self, face: Face, out_size=None, border_expand=(0, 0), align=False,
                      maintain_proportion=False, masked=False):
        # if not specified otherwise, we want to make sure extracted face size
        # is exactly as input face size
        if not out_size:
            out_size = face.get_face_size()

        face.landmarks = self.get_landmarks(face)
        if masked:
            mask = utils.get_face_mask(face, 'hull',
                                       erosion_size=literal_eval(self.config['extract'].get('dilation_kernel', 'None')),
                                       dilation_kernel=literal_eval(self.config['extract'].get('dilation_kernel',
                                                                                               'None')),
                                       blur_size=int(self.config['extract']['blur_size']))
            # black all pixels outside the mask
            face.img = cv2.bitwise_and(face.img, face.img, mask=mask[:, :, 1])
        if align:
            # cut_face = utils.align_face(face, None)
            cut_face = utils._align_face(face, size=out_size)
        else:
            # keep proportions of original image (rect) for extracted image, otherwise resize might stretch the content
            if maintain_proportion:
                border_delta = self._get_maintain_proportion_delta(face.get_face_size(), out_size)
                print(face.get_face_size())
                border_expand = (border_expand[0] + int(border_delta[0]//2), border_expand[1] + int(border_delta[1]//2))
            top, right, bottom, left = (face.rect.top(), face.rect.right(), face.rect.bottom(), face.rect.left())
            x, y = left, top
            w = right - left
            h = bottom - top
            cut_face = face.img[max(0, y-border_expand[1]): y + h + border_expand[1],
                                max(0, x-border_expand[0]): x + w + border_expand[0]]

        cut_face = cv2.resize(cut_face, out_size, interpolation=cv2.INTER_CUBIC)
        return cut_face

    def _get_maintain_proportion_delta(self, src_size, dest_size):
        """
        Return delta amount to maintain destination proportion given source size.
        Tuples order is (w, h)
        :param base_border:
        :param src_size:
        :param dest_size:
        :return:
        """
        dest_ratio = max(dest_size) / min(dest_size)
        print(dest_ratio)
        delta_h = delta_w = 0
        w, h = src_size
        if w > h:
            delta_h = w * dest_ratio - h
        else:
            delta_w = h * dest_ratio - w
        return delta_w, delta_h
