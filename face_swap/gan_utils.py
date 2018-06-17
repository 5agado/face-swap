import keras.backend as K
from keras.layers import concatenate, Lambda
import tensorflow as tf
from tensorflow.contrib.distributions import Beta
from keras.models import Sequential, Model


def cycle_variables(gen):
    """
    Return basic generator components (inputs, outputs and generation function)
    :param gen: generator
    :return:
    """
    distorted_input = gen.inputs[0]
    fake_output = gen.outputs[0]
    fun_generate = K.function([distorted_input], [fake_output])
    return distorted_input, fake_output, fun_generate


def cycle_variables_masked(gen):
    """
    Return masked generator components (inputs, outputs and generation function)
    :param gen: generator
    :return:
    """
    # input and output of the generator
    distorted_input = gen.inputs[0]
    fake_output = gen.outputs[0]

    # in the generator we pre-append an alpha component to the output
    # we here separate such alpha component from the actual bgr image
    alpha = Lambda(lambda x: x[:, :, :, :1])(fake_output)
    rgb = Lambda(lambda x: x[:, :, :, 1:])(fake_output)

    masked_fake_output = alpha * rgb + (1 - alpha) * distorted_input

    fun_generate = K.function([distorted_input], [masked_fake_output])
    fun_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
    fun_abgr = K.function([distorted_input], [concatenate([alpha, rgb])])

    return distorted_input, fake_output, alpha, fun_generate, fun_mask, fun_abgr


def define_loss(netD, real, fake, vggface_feat=None, mixup_alpha=None, use_lsgan=True):
    loss_fn = get_loss_fun(use_lsgan)

    if mixup_alpha:
        dist = Beta(mixup_alpha, mixup_alpha)
        lam = dist.sample()
        mixup = lam * real + (1 - lam) * fake
        output_mixup = netD(mixup)
        loss_D = loss_fn(output_mixup, lam * K.ones_like(output_mixup))
        output_fake = netD(fake)  # dummy
        loss_G = .5 * loss_fn(output_mixup, (1 - lam) * K.ones_like(output_mixup))
    else:
        output_real = netD(real)  # positive sample
        output_fake = netD(fake)  # negative sample
        loss_D_real = loss_fn(output_real, K.ones_like(output_real))
        loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
        loss_D = loss_D_real + loss_D_fake
        loss_G = .5 * loss_fn(output_fake, K.ones_like(output_fake))
    loss_G += K.mean(K.abs(fake - real))

    if not vggface_feat is None:
        loss_G = add_perceptual_loss(loss_G, real=real, fake=fake, vggface_feat=vggface_feat)

    return loss_D, loss_G


def define_loss_masked(netD, real, fake_argb, distorted, vggface_feat=None, mixup_alpha=None, use_lsgan=True):
    # loss weights
    w_D = 0.5  # Discriminator contribution to generator loss
    w_recon = 1.  # L1 reconstruction loss
    w_edge = 1.  # edge loss

    loss_fn = get_loss_fun(use_lsgan)

    alpha = Lambda(lambda x: x[:, :, :, :1])(fake_argb)
    fake_rgb = Lambda(lambda x: x[:, :, :, 1:])(fake_argb)
    fake = alpha * fake_rgb + (1 - alpha) * distorted

    if mixup_alpha:
        dist = Beta(mixup_alpha, mixup_alpha)
        lam = dist.sample()
        mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
        output_mixup = netD(mixup)
        loss_D = loss_fn(output_mixup, lam * K.ones_like(output_mixup))
        output_fake = netD(concatenate([fake, distorted]))  # dummy
        loss_G = w_D * loss_fn(output_mixup, (1 - lam) * K.ones_like(output_mixup))
    else:
        output_real = netD(concatenate([real, distorted]))  # positive sample
        output_fake = netD(concatenate([fake, distorted]))  # negative sample
        loss_D_real = loss_fn(output_real, K.ones_like(output_real))
        loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
        loss_D = loss_D_real + loss_D_fake
        loss_G = w_D * loss_fn(output_fake, K.ones_like(output_fake))

    # Reconstruction loss
    loss_G += w_recon * K.mean(K.abs(fake_rgb - real))

    # Edge loss (similar with total variation loss)
    loss_G += w_edge * K.mean(K.abs(first_order(fake_rgb, axis=1) - first_order(real, axis=1)))
    loss_G += w_edge * K.mean(K.abs(first_order(fake_rgb, axis=2) - first_order(real, axis=2)))

    # Perceptual Loss
    if not vggface_feat is None:
        loss_G = add_perceptual_loss_masked(loss_G, real=real, fake=fake, vggface_feat=vggface_feat, fake_rgb=fake_rgb)

    return loss_D, loss_G


# Build a perceptual-loss model from VGG pre-trained model
# Via Keras can load VGG model via
# vgg = VGG16(include_top=False, model='resnet50', input_shape=HR_IMG_SHAPE) # model can be 'resnet50' or 'vgg16'
# of for VGGFace
# vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
def perceptual_loss_model(vgg_model=None):
    if vgg_model:
        vgg_model.trainable = False
        out_size55 = vgg_model.layers[36].output
        out_size28 = vgg_model.layers[78].output
        out_size7 = vgg_model.layers[-2].output
        vgg_feat = Model(vgg_model.input, [out_size55, out_size28, out_size7])
        vgg_feat.trainable = False
    else:
        vgg_feat = None
    return vgg_feat


def add_perceptual_loss(loss_G, real, fake, vggface_feat):
    pl_params = (0.01, 0.1, 0.1)
    real_sz224 = tf.image.resize_images(real, [224, 224])
    fake_sz224 = tf.image.resize_images(fake, [224, 224])
    real_feat55, real_feat28, real_feat7 = vggface_feat(real_sz224)
    fake_feat55, fake_feat28, fake_feat7 = vggface_feat(fake_sz224)
    loss_G += pl_params[0] * K.mean(K.abs(fake_feat7 - real_feat7))
    loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
    loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))

    return loss_G


def add_perceptual_loss_masked(loss_G, real, fake, vggface_feat, fake_rgb):
    w_pl1 = (0.01, 0.1, 0.2, 0.02)  # perceptual loss 1
    w_pl2 = (0.005, 0.05, 0.1, 0.01)  # perceptual loss 2

    def preprocess_vggface(x):
        x = (x + 1) / 2 * 255  # channel order: BGR
        # x[..., 0] -= 93.5940
        # x[..., 1] -= 104.7624
        # x[..., 2] -= 129.
        x -= [91.4953, 103.8827, 131.0912]
        return x

    pl_params = w_pl1
    real_sz224 = tf.image.resize_images(real, [224, 224])
    real_sz224 = Lambda(preprocess_vggface)(real_sz224)

    # Perceptual loss for masked output
    fake_sz224 = tf.image.resize_images(fake, [224, 224])
    fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
    real_feat112, real_feat55, real_feat28, real_feat7 = vggface_feat(real_sz224)
    fake_feat112, fake_feat55, fake_feat28, fake_feat7 = vggface_feat(fake_sz224)
    loss_G += pl_params[0] * K.mean(K.abs(fake_feat7 - real_feat7))
    loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
    loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))
    loss_G += pl_params[3] * K.mean(K.abs(fake_feat112 - real_feat112))

    # Perceptual loss for raw output
    pl_params = w_pl2
    fake_sz224 = tf.image.resize_images(fake_rgb, [224, 224])
    fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
    fake_feat112, fake_feat55, fake_feat28, fake_feat7 = vggface_feat(fake_sz224)
    loss_G += pl_params[0] * K.mean(K.abs(fake_feat7 - real_feat7))
    loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
    loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))
    loss_G += pl_params[3] * K.mean(K.abs(fake_feat112 - real_feat112))

    return loss_G


def first_order(x, axis=1):
    img_nrows = x.shape[1]
    img_ncols = x.shape[2]
    if axis == 1:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    elif axis == 2:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    else:
        return None


def get_loss_fun(use_lsgan=True):
    # least square loss
    if use_lsgan:
        loss_fn = lambda output, target: K.mean(K.abs(K.square(output - target)))
    else:
        loss_fn = lambda output, target: -K.mean(
            K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))
    return loss_fn
