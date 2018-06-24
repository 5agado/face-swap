import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from numpy.random import shuffle
from IPython.display import clear_output


# data-science-utils
from utils import image_processing

from face_swap.plot_utils import plot_sample
from face_swap import FaceGenerator
from face_swap import autoencoder, gan, gan_utils
from face_swap import CONFIG_PATH


def get_original_data(a_images_path, b_images_path, img_size=(256, 256),
                      tanh_fix=False):
    if tanh_fix:
        convert_fun = lambda x: x / 255 * 2 - 1
    else:
        convert_fun = lambda x: x.astype('float32') / 255.

    images_a = image_processing.load_data(image_processing.get_imgs_paths(a_images_path),
                                          img_size, convert_fun=convert_fun)
    images_b = image_processing.load_data(image_processing.get_imgs_paths(b_images_path),
                                          img_size, convert_fun=convert_fun)

    #images_a += images_b.mean(axis=(0, 1, 2)) - images_a.mean(axis=(0, 1, 2))

    return images_a, images_b


# TODO better batching mechanism
# use yield and exhaustive run over data before reshuffling
# ??also option of loading images here instead of all at start (and get list of paths instead)
# TODO more explicit relation between warp multiplication factor and training images input size
def get_training_data(images, batch_size, config, warp_mult_factor=1):
    warped_images = []
    target_images = []

    indexes = np.random.randint(len(images), size=batch_size)
    for index in indexes:
        image = images[index]
        image = FaceGenerator.random_transform(image, **config['random_transform'])
        warped_img, target_img = FaceGenerator.random_warp(image, mult_f=warp_mult_factor)

        warped_images.append(warped_img)
        target_images.append(target_img)

    return np.array(warped_images), np.array(target_images)


def main(_):
    parser = argparse.ArgumentParser(description='Face Swap. Models Training')

    parser.add_argument('-a', metavar='a_images_path', dest='a_images_path')
    parser.add_argument('-b', metavar='b_images_path', dest='b_images_path')
    parser.add_argument('-model_name', metavar='model_name', dest='model_name',
                        default='base_autoencoder')
    parser.add_argument('-model_version', metavar='model_version', dest='model_version',
                        default='v1')
    parser.add_argument('-c', metavar='config_path', dest='config_path',
                        default=CONFIG_PATH)

    args = parser.parse_args()
    a_images_path = Path(args.a_images_path)
    b_images_path = Path(args.b_images_path)
    config_path = Path(args.config_path)
    model_name = args.model_name
    model_version = args.model_version

    with open(str(config_path), 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    model_cfg = cfg[model_name][model_version]
    models_path = model_cfg.get('models_path', None)

    if not a_images_path.exists():
        print("No such path: {}".format(a_images_path))
        sys.exit(1)
    if not b_images_path.exists():
        print("No such path: {}".format(b_images_path))
        sys.exit(1)
    if models_path and not models_path.exists():
        print("No such path for models: {}".format(models_path))
        sys.exit(1)

    vggface = None
    netGA, netGB, netDA, netDB = gan.get_gan(model_cfg)

    # define generation and plotting function
    # depending if using masked gan model or not
    if model_cfg['masked']:
        netGA_train, netGB_train, netDA_train, netDB_train = gan.build_training_functions_masked(
            cfg[model_name][model_version],
            netGA, netGB, netDA, netDB,
            vggface)
        distorted_A, fake_A, mask_A, path_A, fun_mask_A, fun_abgr = gan_utils.cycle_variables_masked(netGA)
        distorted_B, fake_B, mask_B, path_B, fun_mask_B, fun_abgr = gan_utils.cycle_variables_masked(netGB)
        gen_plot_a = lambda x: np.array(path_A([x])[0])
        gen_plot_b = lambda x: np.array(path_B([x])[0])
        gen_plot_mask_a = lambda x: np.array(fun_mask_A([x])[0]) * 2 - 1
        gen_plot_mask_b = lambda x: np.array(fun_mask_B([x])[0]) * 2 - 1
    else:
        netGA_train, netGB_train, netDA_train, netDB_train = gan.build_training_functions(
            cfg[model_name][model_version],
            netGA, netGB, netDA, netDB,
            vggface)
        gen_plot_a = lambda x: netGA.predict(x)
        gen_plot_b = lambda x: netGB.predict(x)

    # Load data
    images_a, images_b = get_original_data(a_images_path, b_images_path)
    samples_a, samples_b = get_original_data(a_images_path, b_images_path, (64, 64))

    errsGA = []
    errsGB = []
    errsDA = []
    errsDB = []

    NB_EPOCH_LOG = 1000
    NB_EPOCH_CHECKPOINT = 2000

    show_plot = True
    batch_size = 32
    total_epochs = 0
    nb_epochs = 10000
    for gen_iterations in tqdm(range(nb_epochs)):
        total_epochs += 1

        warped_A, target_A = get_training_data(images_a, batch_size, cfg)
        warped_B, target_B = get_training_data(images_b, batch_size, cfg)

        # Train discriminators for one batch
        # if gen_iterations % 1 == 0:
        errDA = netDA_train([warped_A, target_A])
        errDB = netDB_train([warped_B, target_B])
        errsDA.append(errDA[0])
        errsDB.append(errDB[0])

        # Train generators for one batch
        errGA = netGA_train([warped_A, target_A])
        errGB = netGB_train([warped_B, target_B])
        errsGA.append(errGA[0])
        errsGB.append(errGB[0])

        if gen_iterations % NB_EPOCH_CHECKPOINT == 0:
            print("Loss_DA: {} Loss_DB: {} Loss_GA: {} Loss_GB: {}".format(errDA, errDB, errGA, errGB))

            # get new batch of images and generate results for visualization
            shuffle(samples_a)
            shuffle(samples_b)

            if show_plot:
                if gen_iterations % (3 * NB_EPOCH_CHECKPOINT) == 0:
                    clear_output()
                plot_sample(samples_a, samples_b,
                            gen_plot_a, gen_plot_b,
                            tanh_fix=True)
                if model_cfg['masked']:
                    plot_sample(samples_a, samples_b,
                                gen_plot_mask_a, gen_plot_mask_b,
                                tanh_fix=True)
            else:
                sample_img_name = "sample_{}.jpg".format(total_epochs)
                plot_sample(samples_a, samples_b,
                            gen_plot_a, gen_plot_b,
                            tanh_fix=True, save_to=sample_img_name)

            # Save models
            netGA.layers[1].save_weights(str(models_path / "encoder.h5"))
            netGA.layers[2].save_weights(str(models_path / "decoder_A.h5"))
            netGB.layers[2].save_weights(str(models_path / "decoder_B.h5"))
            netDA.save_weights(str(models_path / "netDA.h5"))
            netDB.save_weights(str(models_path / "netDB.h5"))


if __name__ == "__main__":
    main(sys.argv[1:])
