import argparse
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm
import logging

# data-science-utils
from utils import image_processing

from face_swap import CONFIG_PATH
from face_swap.FaceDetector import FaceDetector, FaceSwapException


def main(_=None):
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Face-swap. Extract Faces')

    parser.add_argument('-i', metavar='input_path', dest='input_path', required=True)
    parser.add_argument('-o', metavar='output_path', dest='output_path', required=True)
    parser.add_argument('-c', metavar='config_path', dest='config_path',
                        default=CONFIG_PATH)
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('-s', metavar='step_mod', dest='step_mod', default=1,
                        help="Step module defining which face to actually save")

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    config_path = Path(args.config_path)
    step_mod = int(args.step_mod)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not input_path.exists():
        logging.info("No such path: {}".format(input_path))
        sys.exit(1)
    if not output_path.exists():
        logging.info("Creating output dir: {}".format(output_path))
        output_path.mkdir()
    if not config_path.exists():
        logging.info("No such config file: {}".format(config_path))
        sys.exit(1)

    with open(str(config_path), 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    face_detector = FaceDetector(cfg)
    frame_count = 0
    success_count = 0

    # "Load" input video
    input_video = cv2.VideoCapture(str(input_path))

    logging.info("Starting Face Extraction over video")
    # Process frame by frame
    # TODO would be good to have an hint on progress percentage
    while input_video.isOpened():
        ret, frame = input_video.read()
        if ret == True:
            frame_count += 1
            try:
                faces = face_detector.detect_faces(frame)
                for face in faces:
                    extracted_face = face_detector.extract_face(face)

                    if frame_count % step_mod == 0:
                        cv2.imwrite(str(output_path / "face_{:04d}.jpg".format(success_count)),
                                    extracted_face)
                        success_count += 1
            except FaceSwapException as e:
                logging.debug("Frame {}: {}".format(frame_count, e))
            except Exception as e:
                logging.error(e)
                raise
        else:
            break

    # Release everything if job is finished
    input_video.release()

    logging.info("Extracted {}/{} faces".format(success_count, frame_count))


if __name__ == "__main__":
    main(sys.argv[1:])
