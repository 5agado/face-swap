from os.path import join
import numpy as np
import cv2

from face_swap.Face import Face

# Option of relying on MoviePy (http://zulko.github.io/moviepy/index.html)


def convert_video_to_video(video_path: str, out_path: str, frame_edit_fun,
                  codec='mp4v'):
    # "Load" input video
    input_video = cv2.VideoCapture(video_path)

    # Match source video features.
    fps = input_video.get(cv2.CAP_PROP_FPS)
    size = (
        int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(out_path, fourcc, fps, size)

    # Process frame by frame
    #TODO would be good to have an hint on progress percentage
    while input_video.isOpened():
        ret, frame = input_video.read()
        if ret == True:
            frame = frame_edit_fun(frame)
            out.write(frame)
        else:
            break

    # Release everything if job is finished
    input_video.release()
    out.release()


class VideoConverter:
    def __init__(self, use_kalman_filter=False):
        if use_kalman_filter:
            self.bbox_mavg_coef = None
            self.noise_coef = 5e-3  # Increase by 10x if tracking is slow.
            self.kf0 = kalmanfilter_init(self.noise_coef)
            self.kf1 = kalmanfilter_init(self.noise_coef)
        else:
            self.bbox_mavg_coef = 0.35

        self.prev_coords = (0, 0, 0, 0)
        self.frames = 0

    def get_center(self, face: Face):
        rect = face.rect
        coords = (rect.left(), rect.right(), rect.top(), rect.bottom())

        if self.frames != 0:
            coords = self.get_smoothed_coord(coords, face.img.shape)
            self.prev_coords = coords
        else:
            self.prev_coords = coords
            _ = self.get_smoothed_coord(coords, face.img.shape)
        self.frames += 1
        left, right, top, bottom = coords
        return left + (right - left) // 2, top + (bottom - top) // 2

    def get_smoothed_coord(self, coords: tuple, shape):
        # simply rely on moving average
        if self.bbox_mavg_coef:
            coords = tuple([
                # adjust each coordinate based on coefficies and prev coordinate
                int(self.bbox_mavg_coef * prev_coord + (1 - self.bbox_mavg_coef) * curr_coord)
                for curr_coord, prev_coord in zip(coords, self.prev_coords)
            ])
        # use kalman filter
        else:
            x0, x1, y0, y1 = coords
            prev_x0, prev_y0, prev_x1, prev_y1 = self.prev_coords
            x0y0 = np.array([x0, y0]).astype(np.float32)
            x1y1 = np.array([x1, y1]).astype(np.float32)
            if self.frames == 0:
                for i in range(200):
                    self.kf0.predict()
                    self.kf1.predict()
            self.kf0.correct(x0y0)
            pred_x0y0 = self.kf0.predict()
            self.kf1.correct(x1y1)
            pred_x1y1 = self.kf1.predict()
            x0 = np.max([0, pred_x0y0[0][0]]).astype(np.int)
            x1 = np.min([shape[0], pred_x1y1[0][0]]).astype(np.int)
            y0 = np.max([0, pred_x0y0[1][0]]).astype(np.int)
            y1 = np.min([shape[1], pred_x1y1[1][0]]).astype(np.int)
            if x0 == x1 or y0 == y1:
                x0, y0, x1, y1 = prev_x0, prev_y0, prev_x1, prev_y1
            coords = x0, x1, y0, y1
        return coords


def kalmanfilter_init(noise_coef):
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov = noise_coef * np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
    return kf
