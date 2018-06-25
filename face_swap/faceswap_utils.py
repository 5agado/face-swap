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

    for c in range(0, 3):
        # need to check how much I can cover on the destination
        # and make sure source is same size, otherwise throws
        # exception
        dest_shape = res[y1:y2, x1:x2, c].shape[:2]

        alpha_s = mask[:dest_shape[0], :dest_shape[1], c] / 255.0
        alpha_l = 1.0 - alpha_s
        res[y1:y2, x1:x2, c] = (alpha_s * src_img[:dest_shape[0], :dest_shape[1], c] +
                                alpha_l * res[y1:y2, x1:x2, c])

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


def get_align_matrix(from_face: Face, to_face: Face=None):
    # other implementation option see
    # https://matthewearl.github.io/2015/07/28/switching-eds-with-python/

    # TODO check why only 51 landmarks for default aligned face
    # also why specific order for given landmarks

    if to_face:
        return _umeyama(to_face.landmarks, from_face.landmarks, True)[:2]
    else:

        from_face_landmarks = np.array([(x - from_face.rect.left, y - from_face.rect.top)
                                        for (x, y) in from_face.landmarks])
        # need to resize default ones to match given head size
        (w, h) = from_face.img.shape[:2]
        scaled_default_landmarks = default_landmarks_2D * np.array([w, h])
        # default aligned face has only 51 landmarks, so we remove
        # first 17 from the given one in order to align
        return _umeyama(scaled_default_landmarks, from_face_landmarks[17:], True)[:2]


def align_face(from_face: Face, to_face: Face=None):
    # TODO check why first to and then from so that it works as expected
    align_matrix = get_align_matrix(from_face, to_face)

    if to_face is None:
        to_face = from_face

    size = to_face.get_face_img().shape[:2][::-1]
    print(align_matrix)
    #align_matrix = align_matrix * (size[1] - 2 * 48)
    #align_matrix = align_matrix * (size[1] - 2 * 48)
    align_matrix[:, 2] += 48

    from_image_aligned = cv2.warpAffine(from_face.img,
                                        align_matrix,
                                        size,
                                        dst=np.ones(to_face.img.shape,
                                                    dtype=to_face.img.dtype) * 255,
                                        borderMode=cv2.BORDER_TRANSPARENT,
                                        flags=cv2.WARP_INVERSE_MAP)

    return from_image_aligned


def _align_face(face: Face, desired_lx_eye=(0.35, 0.35), size=None):
    if size:
        desired_face_width, desired_face_height = size
    else:
        desired_face_width, desired_face_height = face.get_face_size()

    eyes_center, angle, scale = get_rotation_info(face, desired_face_width, desired_lx_eye)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # update the translation component of the matrix
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_lx_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    #print(M)

    # apply the affine transformation
    (w, h) = (desired_face_width, desired_face_height)
    output = cv2.warpAffine(face.img, M, (w, h),
                            flags=cv2.INTER_CUBIC)
    return output


def get_rotation_info(face: Face, desired_face_width=None, desired_lx_eye=(0.35, 0.35)):
    from face_swap.FaceDetector import FaceDetector
    lx_eye, rx_eye = FaceDetector.get_eyes(face)

    if not desired_face_width:
        desired_face_width, _ = face.get_face_size()

    # compute eye centroids
    lx_eye_center = lx_eye.mean(axis=0).astype("int")
    rx_eye_center = rx_eye.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rx_eye_center[1] - lx_eye_center[1]
    dX = rx_eye_center[0] - lx_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desired_rx_eyeX = 1.0 - desired_lx_eye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_rx_eyeX - desired_lx_eye[0])
    desired_dist *= desired_face_width
    scale = desired_dist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyes_center = ((lx_eye_center[0] + rx_eye_center[0]) // 2,
                   (lx_eye_center[1] + rx_eye_center[1]) // 2)

    return eyes_center, angle, scale


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
