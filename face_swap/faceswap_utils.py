from functools import reduce
import logging

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from face_swap.Face import Face

from skimage.transform._geometric import _umeyama


def get_face_mask(face: Face, mask_type,
                  erosion_size=None,
                  dilation_kernel=None,
                  blur_size: int = None):
    """
    Return mask of mask_type for the given face.
    :param face:
    :param mask_type:
    :param erosion_size:
    :param dilation_kernel:
    :param blur_size:
    :return:
    """
    if mask_type == 'hull':
        # we can rotate the hull mask obtained from original image
        # or re-detect face from aligned image, and get mask then
        mask = get_hull_mask(face, 255)
    elif mask_type == 'rect':
        face_img = face.get_face_img()
        mask = np.zeros(face_img.shape, dtype=face_img.dtype)+255
    else:
        logging.error("No such mask type: {}".format(mask_type))
        raise Exception("No such mask type: {}".format(mask_type))

    # apply mask modifiers
    if erosion_size:
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erosion_size)
        mask = cv2.erode(mask, erosion_kernel, iterations=1)
    if dilation_kernel:
        mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    if blur_size:
        mask = cv2.blur(mask, (blur_size, blur_size))

    return mask


def get_hull_mask(from_face: Face, fill_val=1):
    """

    :param from_face:
    :param fill_val: generally 1 or 255
    :return:
    """
    mask = np.zeros(from_face.img.shape, dtype=from_face.img.dtype)

    hull = cv2.convexHull(np.array(from_face.landmarks).reshape((-1, 2)).astype(int)).flatten().reshape((
        -1, 2))
    hull = [(p[0], p[1]) for p in hull]

    cv2.fillConvexPoly(mask, np.int32(hull), (fill_val, fill_val, fill_val))

    return mask


def seamless_cloning(hull_to, to_face, img_res):
    # Calculate Mask
    hull8U = [(p[0], p[1]) for p in hull_to]

    mask = np.zeros(to_face.img.shape, dtype=to_face.img.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    # Actual seamless cloning
    r = cv2.boundingRect(np.float32([hull_to]))

    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img_res), to_face.img, mask, center, cv2.NORMAL_CLONE)
    return output


def insert_image_in(src_img, target_image, center, mask):
    """
    Insert/superpose source image into target image, matching source with the given center and applying the given mask
    :param src_img:
    :param target_image:
    :param center:
    :param mask:
    :return:
    """
    res = target_image.copy()
    y1 = max(0, center[1] - src_img.shape[0] // 2)
    y2 = y1 + src_img.shape[0]
    x1 = max(0, center[0] - src_img.shape[1] // 2)
    x2 = x1 + src_img.shape[1]

    # need to check how much I can cover on the destination
    # and make sure source is same size, otherwise throws
    # exception
    dest_shape = res[y1:y2, x1:x2, :].shape[:2]

    alpha_s = mask[:dest_shape[0], :dest_shape[1], :] / 255.0
    alpha_l = 1.0 - alpha_s
    res[y1:y2, x1:x2, :] = (alpha_s * src_img[:dest_shape[0], :dest_shape[1], :] +
                            alpha_l * res[y1:y2, x1:x2, :])

    return res


#################################
#           ALIGNMENT           #
#################################

mean_face_x = np.array([
                        0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                        0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                        0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                        0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                        0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                        0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                        0.553364, 0.490127, 0.42689])
mean_face_y = np.array([
                        0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                        0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                        0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                        0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                        0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                        0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                        0.784792, 0.824182, 0.831803, 0.824182])
default_landmarks_2D = np.stack([mean_face_x, mean_face_y], axis=1)


# other implementation option see
# https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
def align_face(face, boundary_resize_factor=None, invert=False, img=None):

    if img is None:
        face_img = face.get_face_img(boundary_resize_factor=boundary_resize_factor)
    else:
        face_img = img
    src_landmarks = np.array([(x - face.rect.left, y - face.rect.top) for (x, y) in face.landmarks])

    # need to resize default ones to match given head size
    (w, h) = face.get_face_size()
    translation = None
    if boundary_resize_factor:
        img_w, img_h = face_img.shape[:2][::-1]
        translation = (img_w - w, img_h - h)
        #w += translation[0]
        #h += translation[1]
    # w/1.5 h/1.5
    scaled_default_landmarks = np.array([(int(x * w), int(y * h)) for (x, y) in default_landmarks_2D])
    # default aligned face has only 51 landmarks, so we remove
    # first 17 from the given one in order to align
    src_landmarks = src_landmarks[17:]
    target_landmarks = scaled_default_landmarks

    if invert:
        align_matrix = get_align_matrix(target_landmarks, src_landmarks, translation)
    else:
        align_matrix = get_align_matrix(src_landmarks, target_landmarks, translation)

    aligned_img = cv2.warpAffine(face_img,
                                 align_matrix,
                                 (w, h),
                                 borderMode=cv2.BORDER_REPLICATE)

    return aligned_img, align_matrix


def get_align_matrix(src_landmarks, target_landmarks, translation: tuple = None):
    align_matrix = _umeyama(src_landmarks, target_landmarks, True)[:2]

    if translation:
        align_matrix[0, 2] -= translation[0]//2
        align_matrix[1, 2] -= translation[1]//2

    return align_matrix


#################################
#        COLOR CORRECTION
#################################

def correct_colours(img_1, img_2, blur_amount):
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(img_1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(img_2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (img_2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


# See also http://vzaguskin.github.io/histmatching1/
def hist_match(source, template):
    # Code borrow from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def color_hist_match(src_im, tar_im):
    #src_im = cv2.cvtColor(src_im, cv2.COLOR_BGR2Lab)
    #tar_im = cv2.cvtColor(tar_im, cv2.COLOR_BGR2Lab)
    matched_R = hist_match(src_im[:,:,0], tar_im[:,:,0])
    matched_G = hist_match(src_im[:,:,1], tar_im[:,:,1])
    matched_B = hist_match(src_im[:,:,2], tar_im[:,:,2])
    matched = np.stack((matched_R, matched_G, matched_B), axis=2).astype(np.float64)
    return matched


def hist_eq(source, template, nbr_bins=256):
    imres = source.copy()
    for d in range(source.shape[2]):
        imhist, bins = np.histogram(source[:, :, d].flatten(), nbr_bins, normed=True)
        tinthist, bins = np.histogram(template[:, :, d].flatten(), nbr_bins, normed=True)

        cdfsrc = imhist.cumsum() #cumulative distribution function
        cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8) #normalize

        cdftint = tinthist.cumsum() #cumulative distribution function
        cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8) #normalize

        im2 = np.interp(source[:, :, d].flatten(), bins[:-1], cdfsrc)

        im3 = np.interp(im2, cdftint, bins[:-1])

        imres[:, :, d] = im3.reshape((source.shape[0], source.shape[1]))
    return imres
