import argparse
import logging
import sys
from ast import literal_eval
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# data-science-utils
from utils import image_processing
from utils import super_resolution

from face_swap import CONFIG_PATH
from face_swap import FaceGenerator
from face_swap import autoencoder, gan, gan_utils
from face_swap import faceswap_utils as utils
from face_swap import video_utils
from face_swap.Face import Face
from face_swap.FaceDetector import FaceDetector, FaceSwapException
from face_swap.video_utils import VideoConverter


class Swapper:
    def __init__(self, face_detector: FaceDetector, face_generator: FaceGenerator, config, save_all=False):
        """
        Utility object that holds and manages the components necessary for swapping faces in a picture.
        :param face_detector:
        :param face_generator:
        :param config:
        :param save_all:
        """
        self.face_detector = face_detector
        self.face_generator = face_generator
        self.save_all = save_all
        self.config = config

        # Video converter helps smoothing frames transition, via moving average or kalman filter
        self.video_converter = None
        if 'video_smooth_filter' in self.config:
            self.video_converter = video_utils.VideoConverter(use_kalman_filter=self.config['video_smooth_filter'] ==
                                                        'kalman', bbox_mavg_coef=self.config.get('bbox_mavg_coef',
                                                                                                 None))

    def swap(self, from_img, swap_all=True):
        """
        Operate face swap on the given image
        :param from_img: raw image to process
        :param swap_all: whether to swap all faces or just the first detected
        :return:
        """
        try:
            from_faces = self.face_detector.detect_faces(from_img)
        except FaceSwapException as e:
            # reset video converter is we missed a face. Could be a random error or a change or scene
            if self.video_converter:
                self.video_converter.reset_state()
            # if specified save original picture also in case of exceptions
            # this guarantees for example that all frames are present
            if self.save_all:
                logging.debug("Exception: {}. Saving original image".format(e))
                return from_img
            else:
                raise

        # Log advice in case video smoothing in enabled with a multiple faces video
        if len(from_faces) > 1 and self.video_converter:
            logging.info("Multiple faces detected. Suggest to disable video_smooth_filter as it supports only single "
                         "face videos.")

        if swap_all:
            # in case of multiple faces, one option is to save partial results for each face and pass it as original
            # for next face
            part_res = from_img
            for face in from_faces:
                # set previous partial results as img of current face (equivalent to initial img for first face)
                face.img = part_res
                part_res = swap_faces(face, self.face_detector, self.config, self.face_generator,
                                      video_converter=self.video_converter)
            return part_res
        else:
            # otherwise for now just swap first face detected
            return swap_faces(from_faces[0], self.face_detector, self.config, self.face_generator,
                              video_converter=self.video_converter)


def swap_faces(from_face: Face, detector: FaceDetector,
               config, generator: FaceGenerator,
               video_converter: VideoConverter = None):
    """
    Swap from_face face.
    Generate a new face from from_face using generator and blend it in the original image.
    :param from_face:
    :param detector:
    :param config:
    :param generator:
    :param video_converter
    :return:
    """

    if from_face.landmarks is None:
        from_face.landmarks = detector.get_landmarks(from_face)

    # generate a new face from the given one using provided generator
    new_face_img, gen_mask = generator.generate(from_face)
    #return new_face_img

    new_face_landmarks = np.array([(x-from_face.rect.left(), y-from_face.rect.top())
                                   for (x, y) in from_face.landmarks])
    new_face = Face(new_face_img)
    new_face.landmarks = new_face_landmarks

    # process and get high-level mask
    mask = _get_mask(config, from_face, gen_mask)
    #return mask

    # obtain destination face center (absolute to whole image)
    # if define, we rely on a video converter to smooth
    # the center based on previous frames
    #center = from_face.get_face_center()
    #cv2.circle(from_face.img, center, 5, (0, 0, 255), -1)
    if video_converter:
        center = video_converter.get_center(from_face)
    else:
        center = from_face.get_face_center()
    #cv2.circle(from_face.img, center, 5, (255, 0, 0), -1)
    #return from_face.img

    # merge from and to faces via the specified method
    if config.get('seamless_clone'):
        res = cv2.seamlessClone(new_face.img.astype(np.uint8),
                                from_face.img.astype(np.uint8),
                                mask,
                                center[:][::-1],
                                cv2.NORMAL_CLONE)
    else:
        if 'color_correct_blur_frac' in config:
            color_correct_blur_frac = config['color_correct_blur_frac']

            # we compute the blur amount to use based on distance between eyes
            blur_amount = color_correct_blur_frac * np.linalg.norm(
                np.mean(new_face.landmarks[FaceDetector.left_eye], axis=0) -
                np.mean(from_face.landmarks[FaceDetector.right_eye], axis=0))

            # notice that we pass the face images, which need to be of same size
            # for this method to work
            from_face_img = from_face.get_face_img()
            new_face.img = utils.correct_colours(from_face_img, new_face.img, blur_amount)
            #new_face.img = utils.hist_eq(new_face.img, from_face_img)
            #new_face.img = utils.color_hist_match(new_face.img, from_face_img)


        # basic insertion with masking
        res = utils.insert_image_in(new_face.img, from_face.img, center, mask)

    return res


def _get_mask(config, from_face: Face, gen_mask=None):
    mask_method = config.get('mask_method', 'face_mask')
    # if we didn't get a mask directly from the generator, compute it from the face
    if gen_mask is None or mask_method == 'mix_mask' or mask_method == 'face_mask':
        # get base mask
        # better to get from from_face, cause uses all image, and modificators for mask
        # work best if mask itself not close to borders
        face_mask = utils.get_face_mask(from_face, mask_type=config.get('mask_type', 'rect'),
                                        erosion_size=literal_eval(config.get('erosion_size', None)),
                                        blur_size=config.get('blur_size', None))

        # TODO refactor to method
        top, right, bottom, left = (from_face.rect.top(), from_face.rect.right(), from_face.rect.bottom(), from_face.rect.left())
        x, y = left, top
        w = right - left
        h = bottom - top
        border_expand = (0, 0)
        face_mask = face_mask[max(0, y - border_expand[1]): y + h + border_expand[1],
                   max(0, x - border_expand[0]): x + w + border_expand[0], :]

        if gen_mask is not None and mask_method == 'mix_mask':
            mask = np.clip(face_mask + gen_mask, 0, 255)
        else:
            mask = face_mask
    else:
        if mask_method == 'gen_mask_fix':
            blur_size = config.get('blur_size', 1)
            mask = cv2.blur(gen_mask, (blur_size, blur_size))
        else:
            mask = gen_mask
    return mask


def main(_=None):
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Deep-Faceswap. Generative based method')

    parser.add_argument('-i', metavar='input_path', dest='input_path')
    parser.add_argument('-o', metavar='output_path', dest='output_path', required=True)
    parser.add_argument('-c', metavar='config_path', dest='config_path',
                        default=CONFIG_PATH)
    parser.add_argument('-A', dest='save_all', action='store_true',
                        help="Save all images no matter the exception")
    parser.set_defaults(save_all=False)
    parser.add_argument('-model_name', metavar='model_name', dest='model_name',
                        default='base_autoencoder')
    parser.add_argument('-model_version', metavar='model_version', dest='model_version',
                        default='v1')
    parser.add_argument('-process_images', dest='process_images', action='store_true')
    parser.set_defaults(process_images=False)
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    config_path = Path(args.config_path)
    save_all = args.save_all
    process_images = args.process_images
    model_name = args.model_name
    model_version = args.model_version
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not input_path.exists():
        logging.error("No such path: {}".format(input_path))
        sys.exit(1)
    if not output_path.exists() and process_images:
        logging.info("Creating output dir: {}".format(output_path))
        output_path.mkdir()

    # get a valid file from given directory
    if input_path.is_dir() and not process_images:
        video_files = image_processing.get_imgs_paths(input_path, img_types=('*.gif', '*.webm', '*.mp4'), as_str=True)
        if not video_files:
            logging.error("No valid video files in: {}".format(input_path))
            sys.exit(1)
        # for now just pick first one
        input_path = Path(video_files[0])

    with open(str(config_path), 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    face_detector = FaceDetector(cfg)

    model_cfg = cfg[model_name][model_version]

    resize_fun = None
    if cfg['swap']['use_super_resolution']:
        sr_model = super_resolution.get_SRResNet(cfg['super_resolution'])
        resize_fun = lambda img, size: FaceGenerator.super_resolution_resizing(sr_model, img, size)

    # TODO improve check
    # load generators (autoencoder or GAN)
    if "gan" in model_name:
        gen_a, gen_b, _, _ = gan.get_gan(model_cfg, load_discriminators=False)
    else:
        gen_a, gen_b = autoencoder.get_autoencoders(model_cfg)
    target_gen = gen_a if cfg.get('use_aut_a') else gen_b

    gen_input_size = literal_eval(model_cfg['img_shape'])[:2]
    # if we use a masked gan, need to define out generate function (not as simple as call predict)
    if model_cfg['masked']:
        g_in, g_out, alpha, fun_generate, fun_mask, fun_abgr = gan_utils.cycle_variables_masked(target_gen)
        gen_fun = lambda x: fun_abgr([np.expand_dims(x, 0)])[0][0]
        face_generator = FaceGenerator.FaceGenerator(
            lambda face_img: FaceGenerator.gan_masked_generate_face(gen_fun, face_img),
            input_size=gen_input_size, config=cfg['swap'], resize_fun=resize_fun)

    # otherwise proceed by simply passing the generator
    else:
        face_generator = FaceGenerator.FaceGenerator(
            lambda face_img: FaceGenerator.aue_generate_face(target_gen, face_img),
            input_size=gen_input_size, config=cfg['swap'], resize_fun=resize_fun)

    swapper = Swapper(face_detector, face_generator, cfg['swap'], save_all=save_all)

    # process directly a video
    if not process_images:
        logging.info("Running Face Swap over video with face generation")
        try:
            frame_edit_fun = lambda x: swapper.swap(x)
            video_utils.convert_video_to_video(str(input_path), str(output_path),
                                      frame_edit_fun)
        except Exception as e:
            logging.error(e)
            raise
    # or process a list of images (e.g. extracted frames)
    else:
        # collected all image paths
        from_img_paths = image_processing.get_imgs_paths(input_path, as_str=False)

        # iterate over all collected image paths
        logging.info("Running Face Swap over images with face generation")
        for from_face in tqdm(from_img_paths):
            from_filename = from_face.name
            #logging.debug("## From {}".format(from_filename))
            res_path = str(output_path / '{}.png'.format(from_filename.split('.')[0]))
            try:
                from_img = cv2.imread(str(input_path / from_filename))
                results = swapper.swap(from_img)
                cv2.imwrite(res_path, results)
            except Exception as e:
                logging.error(e)
                raise


if __name__ == "__main__":
    main(sys.argv[1:])
